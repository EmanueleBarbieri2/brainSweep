import os
os.environ["q"] = "true" # Fix for macOS numpy issue

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from dataloader import create_dataloaders
from model import ContrastiveModel

# CONFIG
CONFIG = {
    'batch_size': 16,
    'hidden_dim': 64,
    'embed_dim': 128,
    'dropout': 0.1,
    'num_gnn_layers': 2,
    'projection_layers': 3,
    'activation': 'relu',
    'dti_keep': 0.8,    
    'fmri_keep': 0.5,   
    'num_classes': 2,   # CHANGED: Binary (Healthy vs Sick)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_sweep_model.pth'
}

class FineTuneModel(nn.Module):
    def __init__(self, pre_trained_model, num_classes):
        super().__init__()
        self.encoder = pre_trained_model
        
        # We fuse DTI (128) + fMRI (128) = 256 features
        input_dim = CONFIG['embed_dim'] * 2
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, dti, fmri):
        # 1. Get Embeddings (Allow Gradients!)
        z_dti, z_fmri = self.encoder(
            dti, fmri, 
            dti_keep=CONFIG['dti_keep'], 
            fmri_keep=CONFIG['fmri_keep'],
            use_batch_centering=False 
        )
        
        # 2. Concatenate
        combined = torch.cat([z_dti, z_fmri], dim=1)
        
        # 3. Classify
        return self.head(combined)

def train_finetuning():
    print("--- 1. Setup Binary Fine-Tuning (Healthy vs PD) ---")
    train_loader, val_loader = create_dataloaders(CONFIG['batch_size'], node_drop_prob=0.05)
    
    sample, _ = next(iter(train_loader))
    num_node_features = sample.x.shape[1]
    num_nodes = sample.num_nodes // sample.num_graphs

    # A. Initialize Base Model
    base_model = ContrastiveModel(
        num_nodes=num_nodes, num_node_features=num_node_features,
        embed_dim=CONFIG['embed_dim'], hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout'], num_gnn_layers=CONFIG['num_gnn_layers'],
        projection_layers=CONFIG['projection_layers'], activation=CONFIG['activation']
    ).to(CONFIG['device'])
    
    # B. Load Pre-Trained Weights
    try:
        base_model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("✅ Loaded Pre-trained Weights")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # C. Create Full Model
    model = FineTuneModel(base_model, CONFIG['num_classes']).to(CONFIG['device'])
    
    # D. Optimizer 
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5}, # Slow updates for GNN
        {'params': model.head.parameters(), 'lr': 1e-3}     # Fast updates for Classifier
    ], weight_decay=1e-4)

    # Weighted Loss (Optional: if classes are imbalanced)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 

    print("\n--- 2. Fine-Tuning Loop ---")
    best_val_f1 = 0.0
    
    for epoch in range(50): 
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_dti, batch_fmri in train_loader:
            batch_dti, batch_fmri = batch_dti.to(CONFIG['device']), batch_fmri.to(CONFIG['device'])
            
            # --- MERGE LABELS (The Fix) ---
            # Set Class 2 (SWEDD) to Class 1 (PD)
            batch_dti.y[batch_dti.y == 2] = 1 
            # ------------------------------

            optimizer.zero_grad()
            logits = model(batch_dti, batch_fmri)
            loss = criterion(logits, batch_dti.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_dti.y).sum().item()
            total += batch_dti.y.size(0)

        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_dti, batch_fmri in val_loader:
                batch_dti, batch_fmri = batch_dti.to(CONFIG['device']), batch_fmri.to(CONFIG['device'])
                
                # --- MERGE LABELS ---
                batch_dti.y[batch_dti.y == 2] = 1 
                # --------------------

                logits = model(batch_dti, batch_fmri)
                pred = logits.argmax(dim=1)
                
                val_correct += (pred == batch_dti.y).sum().item()
                val_total += batch_dti.y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch_dti.y.cpu().numpy())
        
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # SAVE TO A NEW FILE
            torch.save(model.state_dict(), "best_finetuned_model.pth")
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.3f} | Train: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.3f} >>> SAVED to best_finetuned_model.pth")
        else:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.3f} | Train: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.3f}")

    print(f"\nFinal Best Val F1: {best_val_f1:.3f}")
    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Healthy', 'PD']))

if __name__ == "__main__":
    train_finetuning()