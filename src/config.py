from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Config:
    do_binarization: bool = True
    batch_size: int = 128
    lr: float = 1e-2
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    
    epochs_finetune: int = 50
    epochs_binarize: int = 150
    
    d_out: int = 100
    p_min: float = 0.0
    p_max: float = 1.0
    bin_ratio_max: float = 1.0
    ste_in_bin: bool = True
    use_bin_loss: bool = False
    bin_loss_lambda_max: float = 0.01

    fastbin_end_ratio: float = 0.5
    quant_bits: int = 8

    # Training Toggles
    use_amp: bool = True
    label_smoothing: float = 0.1
    grad_clip_max_norm: float = 1.0

    # Augmentation
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 1.0
    mixup_cutmix_prob_max: float = 1.0
    mixup_switch_prob: float = 0.5

    # Schedules
    schedule_phases_lr: list = field(default_factory=lambda: [
        [0.05, "linear", 1e-5, 1e-2], 
        [1.0, "cosine", 1e-2, 1e-5]
    ])
    schedule_phases_bin_ratio: list = field(default_factory=lambda: [
        [0.30, "cosine", 0.0, 0.5],
        [1.0, "cosine", 0.5, 1.0]
    ])
    schedule_phases_bin_loss_lambda: list = field(default_factory=lambda: [
        [0.5, "stay", 0.0, 0.0],
        [1.0, "linear", 0.0, 0.01]
    ])
    schedule_phases_p: list = field(default_factory=lambda: [
        [0.30, "cosine", 0.0, 0.5],
        [1.0, "cosine", 0.5, 1.0]
    ])
    schedule_phases_mixup_cutmix_prob: list = field(default_factory=lambda: [
        [1.0, "stay", 1.0, 1.0]
    ])