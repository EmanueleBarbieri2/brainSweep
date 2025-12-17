import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import wandb
import os
import argparse
import random
import numpy as np

from dataloader import create_dataloaders
from model import ContrastiveModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_sweep():
    wandb.init(config={
        "batch_size": 32, "epochs": 200, "lr": 0.0003, "weight_decay": 1e-5,
        "hidden_dim": 64, "embed_dim": 128, "dropout": 0.1, "temperature": 0.07,
        "node_drop_prob": 0.05, "dti_keep_ratio": 0.5, "fmri_keep_ratio": 0.2,
        "num_gnn_layers": 2, "projection_layers": 3, "activation": "relu",
        "use_batch_centering": True
    })
    
    config = wandb.config
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup Data
    train_loader, val_loader = create_dataloaders(config.batch_size, 2, config.node_drop_prob)
    sample_dti, _ = next(iter(train_loader))
    num_node_features = sample_dti.x.shape[1]
    num_nodes = sample_dti.num_nodes // sample_dti.num_graphs

    # Setup Model
    model = ContrastiveModel(
        num_nodes=num_nodes, num_node_features=num_node_features, 
        embed_dim=config.embed_dim, hidden_dim=config.hidden_dim, dropout=config.dropout,
        num_gnn_layers=config.num_gnn_layers, projection_layers=config.projection_layers,
        activation=config.activation
    ).to(device)
    
    model.temperature.data.fill_(config.temperature)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=config.epochs - 10, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

    best_top1, best_mrr = 0.0, 0.0
    patience, counter = 200, 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch_dti, batch_fmri in train_loader:
            batch_dti, batch_fmri = batch_dti.to(device), batch_fmri.to(device)
            z_dti, z_fmri = model(
                batch_dti, batch_fmri, 
                dti_keep=config.dti_keep_ratio, fmri_keep=config.fmri_keep_ratio,
                use_batch_centering=config.use_batch_centering
            )
            logits = (z_dti @ z_fmri.T) / model.temperature
            labels = torch.arange(logits.size(0)).to(device)
            loss = (nn.functional.cross_entropy(logits, labels) + 
                    nn.functional.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        top1, top5, mrr = validate(model, val_loader, device, config.dti_keep_ratio, config.fmri_keep_ratio, config.use_batch_centering)
        
        if top1 > best_top1:
            best_top1, best_mrr, counter = top1, mrr, 0
            torch.save(model.state_dict(), "best_sweep_model.pth")
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | MRR: {mrr:.4f} >>> SAVED BEST")
        else:
            counter += 1
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | MRR: {mrr:.4f} | Patience: {counter}/{patience}")

        wandb.log({"epoch": epoch+1, "loss": avg_loss, "val_top1": top1, "val_top5": top5, "val_mrr": mrr, "best_top1": best_top1})
        if counter >= patience: break

def validate(model, loader, device, dti_keep, fmri_keep, use_batch_centering):
    model.eval()
    dti_embeds, fmri_embeds = [], []
    with torch.no_grad():
        for bd, bf in loader:
            bd, bf = bd.to(device), bf.to(device)
            zd, zf = model(bd, bf, dti_keep=dti_keep, fmri_keep=fmri_keep, use_batch_centering=use_batch_centering)
            dti_embeds.append(torch.nn.functional.normalize(zd, dim=1))
            fmri_embeds.append(torch.nn.functional.normalize(zf, dim=1))
    dti_feats = torch.cat(dti_embeds)
    fmri_feats = torch.cat(fmri_embeds)
    sim_matrix = dti_feats @ fmri_feats.T
    n = sim_matrix.size(0)
    targets = torch.arange(n).to(device).view(-1, 1)
    indices = torch.sort(sim_matrix, dim=1, descending=True).indices
    top1 = (indices[:, 0:1] == targets).sum().item() / n * 100
    top5 = (indices[:, :5] == targets).sum().item() / n * 100
    pos = torch.where(indices == targets)[1] 
    mrr = (1.0 / (pos + 1).float()).mean().item()
    return top1, top5, mrr

if __name__ == "__main__":
    train_sweep()