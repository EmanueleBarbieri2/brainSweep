import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb
import numpy as np
import os

# Import your local files
from dataset import BrainCLIPDataset
from augmentations import GraphAugment
from model import DualGNN_CLIP

# --- HELPER: Handle Booleans for WandB ---
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    return False

# --- LOSS FUNCTIONS ---
def contrastive_loss(fmri_emb, dti_emb, logit_scale):
    """Symmetric InfoNCE Loss."""
    # scale is already exp() from the model
    logits = (fmri_emb @ dti_emb.T) * logit_scale
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_f = nn.CrossEntropyLoss()(logits, labels)
    loss_d = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_f + loss_d) / 2

def hard_negative_loss(fmri_emb, dti_emb, logit_scale, beta=0.5):
    """Weighs hard negatives more heavily."""
    logits = (fmri_emb @ dti_emb.T) * logit_scale
    batch_size = logits.shape[0]
    mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
    
    pos_scores = logits[mask].view(batch_size, 1)
    neg_scores = logits[~mask].view(batch_size, -1)
    
    # Weight negatives
    neg_weights = torch.softmax(neg_scores * beta, dim=1).detach()
    weighted_neg_sum = (neg_weights * torch.exp(neg_scores)).sum(dim=1, keepdim=True)
    
    log_prob = pos_scores - torch.log(torch.exp(pos_scores) + weighted_neg_sum + 1e-6)
    return -log_prob.mean()

def vicreg_loss(z1, z2, lambda_=25.0, mu_=25.0, nu_=1.0):
    loss_inv = F.mse_loss(z1, z2)
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    loss_var = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
    
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov2 = (z2.T @ z2) / (z2.shape[0] - 1)
    loss_cov = (cov1.pow(2).sum() - cov1.diag().pow(2).sum()) + (cov2.pow(2).sum() - cov2.diag().pow(2).sum())
    return lambda_ * loss_inv + mu_ * loss_var + nu_ * loss_cov

def calculate_metrics(fmri_emb, dti_emb, top_k=5):
    """Calculates Top-1, Top-K, and Mean Reciprocal Rank (MRR)."""
    logits = fmri_emb @ dti_emb.T
    batch_size = logits.shape[0]
    
    # Sort scores descending
    # indices: (B, B)
    _, indices = torch.sort(logits, dim=1, descending=True)
    
    # Ground truth is simply 0, 1, 2... since diagonal is correct
    targets = torch.arange(batch_size, device=logits.device).view(-1, 1)
    
    # Find where the target is in the sorted list
    # ranks will be 0 for Top-1, 1 for Rank 2, etc.
    ranks = (indices == targets).nonzero(as_tuple=True)[1]
    
    # Metrics
    top1 = (ranks == 0).float().mean().item() * 100
    topk = (ranks < top_k).float().mean().item() * 100
    mrr = (1.0 / (ranks.float() + 1.0)).mean().item()
    
    return top1, topk, mrr

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--gnn_depth', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--gat_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--loss_type', type=str, default='hard_neg', choices=['clip', 'vicreg', 'hard_neg'])
parser.add_argument('--hard_neg_beta', type=float, default=2.0)
parser.add_argument('--aug_mode', type=str, default='mix')
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--jitter', type=float, default=0.05)
parser.add_argument('--sparsity', type=float, default=0.0)
parser.add_argument('--use_curriculum', type=str2bool, default=False)
parser.add_argument('--tta_steps', type=int, default=5)
parser.add_argument('--project_name', type=str, default="brain-clip-final")
args = parser.parse_args()

# PATHS
TRAIN_FMRI_DIR = "./data/train/matrices/fMRI_FC"
TRAIN_DTI_DIR  = "./data/train/matrices/DTI_SC"
VAL_FMRI_DIR   = "./data/val/matrices/fMRI_FC"
VAL_DTI_DIR    = "./data/val/matrices/DTI_SC"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    wandb.init(project=args.project_name, config=vars(args))
    cfg = wandb.config

    print(f"--- Weighted Skip GNN Training (Stable) ---")
    
    aug = GraphAugment(mode=cfg.aug_mode, drop_prob=cfg.drop_prob, jitter_sigma=cfg.jitter, threshold=cfg.sparsity)

    try:
        train_dataset = BrainCLIPDataset(TRAIN_FMRI_DIR, TRAIN_DTI_DIR, fmri_transform=aug, dti_transform=aug)
        val_dataset = BrainCLIPDataset(VAL_FMRI_DIR, VAL_DTI_DIR)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    except Exception as e:
        print(f"Data Error: {e}")
        return

    model = DualGNN_CLIP(
        num_nodes=90, 
        hidden_dim=cfg.hidden_dim, 
        out_dim=cfg.embed_dim,
        num_layers=cfg.gnn_depth,
        heads=cfg.gat_heads,
        dropout=cfg.dropout
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        # Curriculum Learning
        if cfg.use_curriculum:
            progress = min(1.0, epoch / (cfg.epochs * 0.7))
            curr_sparsity = max(0.0, cfg.sparsity * (1.0 - progress))
            train_dataset.fmri_transform.threshold = curr_sparsity
            train_dataset.dti_transform.threshold = curr_sparsity

        model.train()
        total_loss = 0
        
        for fmri, dti, _ in train_loader:
            fmri, dti = fmri.to(DEVICE), dti.to(DEVICE)
            optimizer.zero_grad()
            
            f_glob, d_glob, scale, _, _ = model(fmri, dti)
            
            if cfg.loss_type == 'clip':
                loss = contrastive_loss(f_glob, d_glob, scale)
            elif cfg.loss_type == 'hard_neg':
                loss = hard_negative_loss(f_glob, d_glob, scale, beta=cfg.hard_neg_beta)
            elif cfg.loss_type == 'vicreg':
                loss = vicreg_loss(f_glob, d_glob)
            
            loss.backward()
            
            # --- STABILITY FIX: GRADIENT CLIPPING ---
            # Prevents the "Spike" that kills training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})

        # Validation (TTA)
        if (epoch + 1) % 5 == 0:
            model.eval()
            tta_aug = GraphAugment(mode='jitter', jitter_sigma=0.02, threshold=0.0)
            
            with torch.no_grad():
                all_f_emb, all_d_emb = [], []
                for _ in range(cfg.tta_steps):
                    step_f, step_d = [], []
                    for fmri, dti, _ in val_loader:
                        fmri, dti = fmri.to(DEVICE), dti.to(DEVICE)
                        if cfg.tta_steps > 1:
                            fmri = tta_aug(fmri)
                            dti = tta_aug(dti)
                        f_e, d_e, _, _, _ = model(fmri, dti)
                        step_f.append(f_e); step_d.append(d_e)
                    all_f_emb.append(torch.cat(step_f)); all_d_emb.append(torch.cat(step_d))
                
                # Average embeddings
                val_f = torch.stack(all_f_emb).mean(dim=0)
                val_d = torch.stack(all_d_emb).mean(dim=0)
                
                # Calculate Detailed Metrics
                top1, top5, mrr = calculate_metrics(val_f, val_d, top_k=5)
                
                wandb.log({"val_top5_acc": top5, "val_top1_acc": top1, "val_mrr": mrr, "epoch": epoch})
                print(f"Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Top-1: {top1:.1f}% | Top-5: {top5:.1f}% | MRR: {mrr:.3f}")
                
                if top5 > best_val_acc:
                    best_val_acc = top5
                    local_path = "best_model.pth"
                    torch.save(model.state_dict(), local_path)
                    if wandb.run is not None:
                        try: wandb.save(local_path, base_path=os.getcwd())
                        except: pass
                    print(f"  >>> Best Saved! ({best_val_acc:.1f}%)")

    wandb.finish()

if __name__ == "__main__":
    main()