import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import numpy as np
import wandb
import timm

from config import Config
from utils import get_scheduled_param, rand_bbox, get_weight_stats, print_weight_stats, param_count
from models import convert_to_bit_model, BitModelWrapper
from dataset import get_dataloaders

def create_base_model(config):
    model = timm.create_model('resnet18', pretrained=True, num_classes=config.d_out)

    # We train on CIFAR100: Replace Stem (tuned for imagenet).
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.maxpool = nn.Identity()
    
    convert_to_bit_model(model)
    return BitModelWrapper(model, use_bin_loss=config.use_bin_loss)

def _train_loop(model, config: Config, trainloader, testloader, epochs, device, is_binarizing, run_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=config.use_amp)
    
    total_steps = len(trainloader) * epochs
    global_step = 0

    wandb.init(project="bit-resnet18-cifar100", name=run_name, config=config.__dict__)
    wandb.watch(model, criterion, log="all", log_freq=100)

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        last_p, last_bin_ratio = 0.0, 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            current_lr = get_scheduled_param(global_step, total_steps, config.schedule_phases_lr)
            for param_group in optimizer.param_groups: param_group['lr'] = current_lr

            current_p, current_bin_ratio, current_bin_loss_lambda = 0.0, 0.0, 0.0
            
            # Use schedules ONLY if we are in the binarizing phase
            if is_binarizing and config:
                current_p = get_scheduled_param(global_step, total_steps, config.schedule_phases_p)
                if config.use_bin_loss:
                    current_bin_loss_lambda = get_scheduled_param(global_step, total_steps, config.schedule_phases_bin_loss_lambda)
                    current_bin_ratio = model.bin_ratio_param.item()
                else:
                    current_bin_ratio = get_scheduled_param(global_step, total_steps, config.schedule_phases_bin_ratio)

            last_p, last_bin_ratio = current_p, current_bin_ratio
            current_mixup_prob = get_scheduled_param(global_step, total_steps, config.schedule_phases_mixup_cutmix_prob)

            # Mixup/Cutmix
            labels_training = labels
            if np.random.rand() < current_mixup_prob:
                labels_one_hot = F.one_hot(labels, num_classes=config.d_out).float()
                if config.label_smoothing > 0:
                    uniform_dist = torch.full_like(labels_one_hot, config.label_smoothing / config.d_out)
                    labels_one_hot = (1.0 - config.label_smoothing) * labels_one_hot + uniform_dist
                
                lam = np.random.beta(config.mixup_alpha, config.mixup_alpha) if np.random.rand() < config.mixup_switch_prob else np.random.beta(config.cutmix_alpha, config.cutmix_alpha)
                rand_idx = torch.randperm(inputs.size(0)).to(device)
                
                if np.random.rand() < config.mixup_switch_prob:
                    inputs = lam * inputs + (1 - lam) * inputs[rand_idx, :]
                else:
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_idx, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                labels_training = lam * labels_one_hot + (1 - lam) * labels_one_hot[rand_idx, :]

            optimizer.zero_grad()
            with autocast("cuda", enabled=config.use_amp):
                outputs = model(inputs, current_p, current_bin_ratio)
                task_loss = criterion(outputs, labels_training)
                total_loss = task_loss - model.bin_ratio_param * current_bin_loss_lambda if config.use_bin_loss else task_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()

            if config.use_bin_loss:
                with torch.no_grad(): model.bin_ratio_param.data.clamp_(0.0, 1.0)

            running_loss += task_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            global_step += 1

        train_acc, avg_train_loss = correct_train / total_train, running_loss / len(trainloader)
        print(f"\n[{run_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Acc: {100*train_acc:.2f}% | BinRatio: {last_bin_ratio:.2f} | P: {last_p:.2f}")

        # Evaluation
        def run_eval(p_val, bin_val):
            model.eval()
            corr, tot, run_loss = 0, 0, 0.0
            with torch.no_grad():
                for imgs, lbls in testloader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    with autocast("cuda", enabled=config.use_amp):
                        outs = model(imgs, p_val, bin_val)
                        run_loss += criterion(outs, lbls).item()
                    _, pred = torch.max(outs.data, 1)
                    tot += lbls.size(0)
                    corr += (pred == lbls).sum().item()
            return corr / tot, run_loss / len(testloader)

        acc_hard, loss_hard = run_eval(1.0, 1.0)
        acc_soft, loss_soft = run_eval(last_p, last_bin_ratio)
        
        log_data = {
            "epoch": epoch + 1, "train/epoch_loss": avg_train_loss, "train/epoch_acc": train_acc,
            "val_hard/accuracy": acc_hard, "val_hard/loss": loss_hard,
            "val_soft/accuracy": acc_soft, "val_soft/loss": loss_soft,
        }
        wandb.log(log_data, step=global_step)

    wandb.finish()
    return model

# def finetune(config: Config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     trainloader, testloader = get_dataloaders(config)
    
#     print("\n--- Starting Phase 1: Finetune (No Binarization) ---")
#     model = create_base_model(config).to(device)
    
#     model = _train_loop(model, config, trainloader, testloader, 
#                         epochs=config.epochs_finetune, device=device, 
#                         is_binarizing=False, run_name="Finetune_Phase")
    
#     os.makedirs("checkpoints", exist_ok=True)
#     torch.save(model.state_dict(), "checkpoints/finetuned.pth")
#     print("Saved finetuned model to checkpoints/finetuned.pth")

# def binarize(config: Config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     trainloader, testloader = get_dataloaders(config)
    
#     ckpt_path = "checkpoints/finetuned.pth"
#     if not os.path.exists(ckpt_path):
#         print("Finetuned checkpoint not found. Falling back to finetuning first...")
#         finetune(config)
        
#     print("\n--- Starting Phase 2: Binarization Pipeline ---")
#     model = create_base_model(config).to(device)
#     model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    
#     _train_loop(model, config, trainloader, testloader, 
#                 epochs=config.epochs_binarize, device=device, 
#                 is_binarizing=True, run_name="Binarize_Phase")

def binarize(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloaders(config)

    print("Creating model ...")    
    model = create_base_model(config).to(device)

    print("Start training ...")    
    _train_loop(model, config, trainloader, testloader, 
                epochs=config.epochs_binarize, device=device, 
                is_binarizing=True, run_name="Binarize_Phase")

    print("Training ended.")    

if __name__ == "__main__":
    cfg = Config()
    binarize(cfg)