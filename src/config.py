from dataclasses import dataclass, field
from typing import List, Any

LR = 1e-3
LR_MIN = 1e-5
NOBIN_FRAC = 0.20 # Fraction of training to stay fully float
FASTBIN_FRAC = 0.30 # We binarize more aggressively until this point
FASTBIN_END_RATIO = 0.5

@dataclass
class Config:
    batch_size: int = 128
    weight_decay: float = 1e-4
    
    epochs_binarize: int = 300
    
    d_out: int = 100
    p_min: float = 0.0
    p_max: float = 1.0
    bin_ratio_max: float = 1.0
    use_bin_loss: bool = False
    bin_loss_lambda_max: float = 0.01

    # Training Toggles
    use_amp: bool = True
    label_smoothing: float = 0.1
    grad_clip_max_norm: float = 1.0

    # Augmentation
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.5
    mixup_cutmix_prob_max: float = 1.0
    mixup_switch_prob: float = 0.5

    schedule_phases_lr: list = field(default_factory=lambda: [
        [0.05, "linear", LR_MIN, LR], 
        [1.0, "cosine", LR, LR_MIN]
    ])
    schedule_phases_bin_ratio: list = field(default_factory=lambda: [
        [NOBIN_FRAC, "stay", 0.0, 0.0],       
        [FASTBIN_FRAC, "cosine", 0.0, FASTBIN_END_RATIO],
        [1.0, "cosine", FASTBIN_END_RATIO, 1.0], 
    ])
    schedule_phases_p: list = field(default_factory=lambda: [
        [NOBIN_FRAC, "stay", 0.0, 0.0],       
        [FASTBIN_FRAC, "cosine", 0.0, FASTBIN_END_RATIO],
        [1.0, "cosine", FASTBIN_END_RATIO, 1.0], 
    ])
    schedule_phases_bin_loss_lambda: list = field(default_factory=lambda: [
        [1.0, "linear", 0.0, 1.0]
    ])
    schedule_phases_mixup_cutmix_prob: list = field(default_factory=lambda: [
        [1.0, "stay", 0.5, 0.5]
    ])