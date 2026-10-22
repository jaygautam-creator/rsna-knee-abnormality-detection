"""
RSNA Knee Abnormality Detection: 3rd Pillar Training Script
Architecture: DINOv2-Base with Anatomical Plane Routing & Multi-Slot Cross-Attention (SlotHead-v2)
Designed to run on Google Colab GPU (T4 / A100 / L4)
Outputs:
  - Checkpoints: /content/checkpoints/dinov2_plane_routed_fold{f}_best.pt
  - OOF Predictions: /content/oof/oof_dinov2_base.csv
  - Spearman Correlation against Raptor & S6
"""

import os
import sys
import time
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

# ---------------------------------------------------------------------------
# Global Configurations
# ---------------------------------------------------------------------------
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

SLOT_NAMES = [
    'Sagittal_FS',     # 0
    'Sagittal_nonFS',  # 1
    'Coronal_FS',      # 2
    'Coronal_nonFS',   # 3
    'Axial_FS',        # 4
    'Axial_nonFS'      # 5
]

ANATOMICAL_PRIOR = {
    'ACL': [0, 1],                    # Sagittal is primary
    'MCL': [2, 3],                    # Coronal is primary
    'Medial Meniscus': [0, 1, 2, 3],  # Sagittal + Coronal
    'Lateral Meniscus': [0, 1, 2, 3], # Sagittal + Coronal
    'Medial OA': [2, 3, 0],           # Coronal + Sagittal
    'Lateral OA': [2, 3, 0],          # Coronal + Sagittal
    'PF OA': [4, 5, 0],               # Axial + Sagittal
    'Effusion': [0, 4, 2],            # Fluid-sensitive (T2/STIR/FS)
    'Synovitis': [0, 4, 2],           # Fluid-sensitive
    "Baker's": [4, 5, 0],             # Axial + Sagittal
    'Contusion': [0, 2, 4],           # Bone marrow edema (STIR/FS)
    'Fracture': [0, 1, 2, 3]          # Cortical disruption + Edema
}

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class AnatomicalSlotHeadV2(nn.Module):
    def __init__(self, embed_dim=768, n_slots=6, n_classes=12, hidden_dim=384, drop_p=0.2):
        super().__init__()
        self.n_slots = n_slots
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        # Intra-series slice attention pooling (16 slices -> 1 slot representation)
        self.slice_attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, 1)
        )

        # Slot feature projection
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_p)
        )

        # Learnable Slot Identifiers
        self.slot_emb = nn.Parameter(torch.randn(1, n_slots, hidden_dim) * 0.02)

        # Finding-Specific Queries
        self.finding_queries = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.02)

        # Anatomical Prior Bias Matrix
        prior_matrix = torch.zeros(n_classes, n_slots, dtype=torch.float32)
        for i, target in enumerate(TARGETS):
            favored_slots = ANATOMICAL_PRIOR.get(target, list(range(n_slots)))
            for slot_idx in favored_slots:
                prior_matrix[i, slot_idx] = 1.5
        self.register_buffer('anatomical_prior', prior_matrix)

        # Dedicated Per-Finding Classifiers
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(drop_p),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_classes)
        ])

    def forward(self, x, slot_mask):
        """
        x: (B, N_SLOTS=6, Slices=16, Embed_Dim)
        slot_mask: (B, N_SLOTS=6)
        """
        B, N_S, S, D = x.shape
        x_flat = x.view(B * N_S, S, D)
        slice_scores = self.slice_attn(x_flat)
        slice_weights = F.softmax(slice_scores, dim=1)
        slot_pooled = torch.sum(slice_weights * x_flat, dim=1)
        slot_features = slot_pooled.view(B, N_S, D)

        h = self.slot_proj(slot_features) + self.slot_emb

        # Finding-to-Slot Cross-Attention
        attn_scores = torch.einsum('bsh,qh->bqs', h, self.finding_queries) / math.sqrt(self.hidden_dim)
        attn_scores = attn_scores + self.anatomical_prior.unsqueeze(0)

        # Zero attention to missing series slots
        mask_expanded = slot_mask.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask_expanded < 0.5, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        finding_contexts = torch.einsum('bqs,bsh->bqh', attn_weights, h)

        logits = []
        for c in range(self.n_classes):
            c_logit = self.classifiers[c](finding_contexts[:, c, :])
            logits.append(c_logit)

        return torch.cat(logits, dim=1), attn_weights


class KneeDINOv2Model(nn.Module):
    def __init__(self, backbone_name='vit_base_patch14_dinov2', pretrained=True,
                 img_size=392, num_slices=16, num_classes=12):
        super().__init__()
        self.img_size = img_size
        self.num_slices = num_slices
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )
        self.embed_dim = self.backbone.num_features
        self.slot_head = AnatomicalSlotHeadV2(
            embed_dim=self.embed_dim,
            n_slots=6,
            n_classes=num_classes,
            hidden_dim=384,
            drop_p=0.2
        )

    def forward(self, study_volume, slot_mask):
        """
        study_volume: (B, N_SLOTS=6, S=16, C=3, H, W)
        slot_mask: (B, N_SLOTS=6)
        """
        B, N_S, S, C, H, W = study_volume.shape
        flat_slices = study_volume.view(B * N_S * S, C, H, W)
        feats = self.backbone(flat_slices)
        slot_feats = feats.view(B, N_S, S, self.embed_dim)
        return self.slot_head(slot_feats, slot_mask)

# ---------------------------------------------------------------------------
# Training & Loss Functions
# ---------------------------------------------------------------------------
class CalibratedFocalLoss(nn.Module):
    """
    Weighted BCE loss with focal modulation for class-imbalanced medical findings
    """
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets, weights):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * bce * weights.unsqueeze(1)
        return loss.mean()

print("[DINOv2 Plane-Routed] Model Architecture and Training Module successfully defined.")
