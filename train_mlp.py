import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import wandb
import numpy as np

# Import your modules
from dataset import BrainCLIPDataset
from augmentations import GraphAugment
from model_mlp import SimpleBrainCLIP  # <--- MLP Model

# --- LOSS FUNCTIONS ---
def contrastive_loss(fmri_emb, dti_emb, logit_scale):
    logits = (fmri_emb @ dti_emb.T) * logit_scale
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_f = nn.CrossEntropyLoss()(logits, labels)
    loss_d = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_f + loss_d) / 2

def vicreg_loss(z1, z2, lambda_=25.0, mu_=25.0, nu_=1.0):
    mse_loss = F.mse_loss(z1, z2)
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov1 = (z1.T @ z1) / (z1.shape[0] - 1)
    cov2 = (z2.T @ z2) / (z2.shape[0] - 1)
    cov_loss = (cov1.pow(2).sum() - cov1.diag().pow(2).sum()) + \
               (cov2.pow(2).sum() - cov2.diag().pow(2).sum())
    return lambda_ * mse_loss + mu_ * std_loss + nu_ * cov_loss

def calculate_accuracy(fmri_emb, dti_emb, top_k=5):
    logits = fmri_emb @ dti_emb.T
    batch_size = logits.shape[0]
    _, pred_indices = logits.topk(top_k, dim=1, largest=True, sorted=True)
    correct = torch.arange(batch_size, device=logits.device).view(-1, 1)
    top1 = (pred_indices[:, 0] == correct.squeeze()).float().sum()
    topk = (pred_indices == correct).any(dim=1).float().sum()
    return top1.item(), topk.item()

# --- ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--loss_type', type=str, default='clip', choices=['clip', 'vicreg'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)

# Augmentations
parser.add_argument('--aug_mode', type=str, default='mix')
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--jitter', type=float, default=0.05)
parser.add_argument('--sparsity', type=float, default=0.0, help="Sparsify threshold (0 to disable)")

# VICReg Params
parser.add_argument('--vicreg_lambda', type=float, default=25.0)
parser.add_argument('--vicreg_mu', type=float, default=25.0)
parser.add_argument('--vicreg_nu', type=float, default=1.0)

args = parser.parse_args()

# PATHS
TRAIN_FMRI = "./data/train/matrices/fMRI_FC"
TRAIN_DTI  = "./data/train/matrices/DTI_SC"
VAL_FMRI   = "./data/val/matrices/fMRI_FC"
VAL_DTI    = "./data/val/matrices/DTI_SC"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    wandb.init(project="brain-clip-mlp", config=vars(args))
    cfg = wandb.config

    print(f"--- BrainCLIP MLP ---")
    print(f"Loss: {cfg.loss_type} | Hidden: {cfg.hidden_dim} | Sparsity: {cfg.sparsity}")

    # 1. Augmentations (Same settings for fMRI/DTI for simplicity, or split if needed)
    aug = GraphAugment(mode=cfg.aug_mode, drop_prob=cfg.drop_prob, jitter_sigma=cfg.jitter, threshold=cfg.sparsity)

    # 2. Dataset
    try:
        train_ds = BrainCLIPDataset(TRAIN_FMRI, TRAIN_DTI, fmri_transform=aug, dti_transform=aug)
        val_ds   = BrainCLIPDataset(VAL_FMRI, VAL_DTI)
    except Exception as e:
        print(f"Data Error: {e}"); return

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 3. Model
    model = SimpleBrainCLIP(
        num_nodes=90, 
        embedding_dim=cfg.embed_dim, 
        hidden_dim=cfg.hidden_dim, 
        dropout=cfg.dropout
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        
        for fmri, dti, _ in train_loader:
            fmri, dti = fmri.to(DEVICE), dti.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward
            f_emb, d_emb, scale = model(fmri, dti)
            
            if cfg.loss_type == 'vicreg':
                loss = vicreg_loss(f_emb, d_emb, cfg.vicreg_lambda, cfg.vicreg_mu, cfg.vicreg_nu)
            else:
                loss = contrastive_loss(f_emb, d_emb, scale)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Log Train
        wandb.log({"train_loss": avg_loss, "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})

        # Validation
        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                f_all, d_all = [], []
                for f, d, _ in val_loader:
                    f, d = f.to(DEVICE), d.to(DEVICE)
                    # Get embeddings
                    f_e, d_e, _ = model(f, d)
                    f_all.append(f_e)
                    d_all.append(d_e)
                
                if f_all:
                    f_cat = torch.cat(f_all); d_cat = torch.cat(d_all)
                    # Correct validation call
                    t1, t5 = calculate_accuracy(f_cat, d_cat, top_k=5)
                    acc_pct = (t5 / len(val_ds)) * 100
                    
                    wandb.log({"val_top5_acc": acc_pct, "epoch": epoch})
                    print(f"Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Val Top-5: {acc_pct:.1f}%")
                    
                    if acc_pct > best_val_acc:
                        best_val_acc = acc_pct
                        torch.save(model.state_dict(), "best_mlp.pth")

    wandb.finish()

if __name__ == "__main__":
    main()