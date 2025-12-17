import torch
from torch_geometric.loader import DataLoader
from dataset import PairedBrainDataset
from transforms import BrainGraphTransform
from augment import GraphAugment

def create_dataloaders(batch_size=32, num_workers=2, node_drop_prob=0.0):
    
    transform = BrainGraphTransform(stats_path='normalization_stats.pt')
    augmentor = GraphAugment(node_drop_prob=node_drop_prob, edge_noise_std=0.02)

    train_dataset = PairedBrainDataset(
        dti_path='data/train/dti_graphs.pt',
        fmri_path='data/train/fmri_graphs.pt',
        transform=transform,
        augment=augmentor
    )
    
    val_dataset = PairedBrainDataset(
        dti_path='data/val/dti_graphs.pt',
        fmri_path='data/val/fmri_graphs.pt',
        transform=transform,
        augment=None
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return train_loader, val_loader