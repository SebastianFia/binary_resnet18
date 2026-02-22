import math
import torch
import numpy as np

def param_count(m):
    return sum(p.numel() for p in m.parameters())

def interp(progress, typ="cosine", start=0.0, end=1.0):
    t = 0
    if typ == "cosine": t = (1 - math.cos(progress * math.pi)) / 2
    elif typ == "linear": t = progress
    elif typ == "stay": t = 0
    return start + (end - start) * t

def get_scheduled_param(current_step, total_steps, schedule_phases):
    progress = min(current_step / total_steps, 1.0)
    i = 0
    for (end_progress, _, _, _) in schedule_phases:
        if progress < end_progress: break
        if i == (len(schedule_phases) - 1): break
        i += 1
    start_progress = 0 if i == 0 else schedule_phases[i - 1][0]
    end_progress, typ, startval, endval = schedule_phases[i]
    if end_progress - start_progress == 0:
        local_progress = 0
    else:
        local_progress = (progress - start_progress) / (end_progress - start_progress)
    return interp(local_progress, typ, startval, endval)

def get_weight_stats(layer):
    w = layer.weight.detach().view(-1)
    mean = w.mean().item()
    close_pct = ((w.abs() - 1.0).abs() < 0.1).float().mean().item() * 100
    return {"mean": mean, "close_pct": close_pct}

def print_weight_stats(layer, name="Layer"):
    stats = get_weight_stats(layer)
    print(f"[{name}] Mean: {stats['mean']:.3f} | Close to ±1: {stats['close_pct']:.1f}%")

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2