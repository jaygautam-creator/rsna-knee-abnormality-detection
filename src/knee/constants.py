"""Competition constants."""

# The 12 findings, in sample_submission.csv column order. The metric is the unweighted mean of their ROC-AUCs.
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

COMPETITION = "rsna-knee-abnormality-detection"
