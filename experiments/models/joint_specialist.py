"""High-Resolution Knee Joint Specialist Architecture.

Implements:
1. Physical Millimeter ROI Cropping (130mm FOV centered at joint line).
2. Laterality Canonicalization (Left knee mirrored so Medial is always Left).
3. Pretrained Feature Extraction with Native AMP (fp16).
4. Multi-Head Slice Attention Pooling over depth slices.
5. Cross-Label Transformer for inter-abnormality biological correlation.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_CONDITIONS = [
    "Lateral Meniscus",
    "Medial Meniscus",
    "PF OA",
    "Synovitis"
]
ALL_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def crop_roi_physical(
    image: np.ndarray,
    pixel_spacing: Optional[float] = None,
    crop_mm: float = 130.0,
    target_size: int = 384
) -> np.ndarray:
    """Extract a physical millimeter field-of-view centered on the knee joint space.
    
    Knee MRIs center the joint line in the scanner bore. Cropping 130mm removes
    peripheral air, muscle, and coil artifacts, preserving native spatial resolution.
    """
    if pixel_spacing is None or pixel_spacing <= 0 or not np.isfinite(pixel_spacing):
        pixel_spacing = 0.45  # Standard clinical knee MRI pixel spacing (~0.45 mm/px)

    h, w = image.shape[-2:]
    want_pixels = int(round(crop_mm / float(pixel_spacing)))

    if 32 < want_pixels < min(h, w):
        cy, cx = h // 2, w // 2
        half = want_pixels // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        cropped = image[..., y1:y2, x1:x2]
    else:
        cropped = image

    # Resize to target resolution if needed
    if cropped.shape[-2] != target_size or cropped.shape[-1] != target_size:
        if cropped.ndim == 2:
            cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        elif cropped.ndim == 3:
            # Slices x H x W
            res = [cv2.resize(cropped[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                   for i in range(cropped.shape[0])]
            cropped = np.stack(res, axis=0)
    return cropped

def canonicalize_laterality(
    volume: np.ndarray,
    plane: str,
    laterality: str
) -> np.ndarray:
    """Canonicalize orientation so Medial is always on the Left side of the image.
    
    For Left knees:
      - Coronal: Flip horizontal axis (W) so Medial compartment is left.
      - Sagittal: Reverse slice depth order so slices proceed from Medial to Lateral.
    """
    is_left = (str(laterality).strip().upper() == 'L')
    if not is_left:
        return volume

    if plane.lower() == 'coronal':
        # Flip width (axis -1)
        return np.flip(volume, axis=-1).copy()
    elif plane.lower() == 'sagittal':
        # Reverse slice ordering (axis 0)
        return np.flip(volume, axis=0).copy()
    return volume

class SliceAttentionPooling(nn.Module):
    """Multi-head cross-attention querying depth slices for each abnormality."""
    
    def __init__(self, embed_dim: int = 768, num_targets: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_targets = num_targets
        self.embed_dim = embed_dim
        
        # Trainable abnormality queries
        self.queries = nn.Parameter(torch.randn(num_targets, embed_dim) * 0.02)
        
        # Multi-head attention across slices
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, slice_feats: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        slice_feats: [Batch, Slices, embed_dim]
        key_padding_mask: [Batch, Slices] (True where padded)
        Returns: [Batch, num_targets, embed_dim]
        """
        B = slice_feats.size(0)
        # Expand queries for batch: [Batch, num_targets, embed_dim]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention: query attends to slice features
        attended, _ = self.mha(
            query=q,
            key=slice_feats,
            value=slice_feats,
            key_padding_mask=key_padding_mask
        )
        out = self.norm(q + attended)
        return out

class CrossLabelTransformer(nn.Module):
    """Models biological co-occurrence patterns between abnormalities."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dim_feedforward: int = 1536, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, label_embeds: torch.Tensor) -> torch.Tensor:
        """
        label_embeds: [Batch, num_targets, embed_dim]
        Returns: [Batch, num_targets, embed_dim]
        """
        out = self.encoder(label_embeds)
        return self.norm(out)

class KneeJointSpecialist(nn.Module):
    """End-to-end Specialist Model with Slice Attention and Cross-Label Interaction."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_targets: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        self.num_targets = num_targets
        self.embed_dim = embed_dim
        
        # Slice Attention Pooling
        self.pooling = SliceAttentionPooling(embed_dim=embed_dim, num_targets=num_targets, dropout=dropout)
        
        # Cross-Label Biological Correlation
        self.cross_label = CrossLabelTransformer(embed_dim=embed_dim, dropout=dropout)
        
        # Per-finding prediction heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            ) for _ in range(num_targets)
        ])
        
    def forward(self, slice_feats: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        slice_feats: [Batch, Slices, embed_dim]
        Returns logits: [Batch, num_targets]
        """
        # 1. Pool slices per finding query: [Batch, num_targets, embed_dim]
        pooled = self.pooling(slice_feats, key_padding_mask=key_padding_mask)
        
        # 2. Inter-label biological interaction: [Batch, num_targets, embed_dim]
        enhanced = self.cross_label(pooled)
        
        # 3. Classify each finding independently
        logits = []
        for i in range(self.num_targets):
            finding_feat = enhanced[:, i, :]
            logit = self.classifiers[i](finding_feat)  # [Batch, 1]
            logits.append(logit)
            
        return torch.cat(logits, dim=-1)  # [Batch, num_targets]

def posterior_calibrated_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    rank_weight: float = 0.15
) -> torch.Tensor:
    """Calculates weighted BCE loss combined with pairwise ranking margin loss."""
    # Weighted Soft Binary Cross Entropy
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weighted_bce = (bce * sample_weights).mean()
    
    if rank_weight <= 0:
        return weighted_bce
        
    # Pairwise ranking loss on high vs low confidence pairs
    diff_targets = targets.unsqueeze(1) - targets.unsqueeze(0)  # [B, B, T]
    diff_logits = logits.unsqueeze(1) - logits.unsqueeze(0)    # [B, B, T]
    
    pos_pairs = (diff_targets > 0.4).float()
    margin_loss = F.relu(0.2 - diff_logits) * pos_pairs
    pair_loss = margin_loss.sum() / (pos_pairs.sum() + 1e-6)
    
    return (1.0 - rank_weight) * weighted_bce + rank_weight * pair_loss
