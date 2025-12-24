import os
os.environ["q"] = "true" 

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
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
    'num_classes': 2,   
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_finetuned_model.pth' 
}

class FineTuneModel(nn.Module):
    def __init__(self, pre_trained_model, num_classes):
        super().__init__()
        self.encoder = pre_trained_model
        input_dim = CONFIG['embed_dim'] * 2
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, dti, fmri):
        z_dti, z_fmri = self.encoder(
            dti, fmri, 
            dti_keep=CONFIG['dti_keep'], 
            fmri_keep=CONFIG['fmri_keep'],
            use_batch_centering=False 
        )
        return self.head(torch.cat([z_dti, z_fmri], dim=1))

def evaluate_model():
    print("--- 1. Loading Model & Data ---")
    _, val_loader = create_dataloaders(CONFIG['batch_size'], node_drop_prob=0.0)
    
    sample, _ = next(iter(val_loader))
    real_nodes = sample.num_nodes // sample.num_graphs
    real_feats = sample.x.shape[1]

    base_model = ContrastiveModel(
        num_nodes=real_nodes, num_node_features=real_feats,
        embed_dim=CONFIG['embed_dim'], hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout'], num_gnn_layers=CONFIG['num_gnn_layers'],
        projection_layers=CONFIG['projection_layers'], activation=CONFIG['activation']
    ).to(CONFIG['device'])
    
    model = FineTuneModel(base_model, CONFIG['num_classes']).to(CONFIG['device'])
    
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("✅ Fine-Tuned Weights Loaded.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = [] # Needed for ROC-AUC

    print("--- 2. Running Inference ---")
    with torch.no_grad():
        for batch_dti, batch_fmri in val_loader:
            batch_dti = batch_dti.to(CONFIG['device'])
            batch_fmri = batch_fmri.to(CONFIG['device'])
            
            # Binary Merge
            batch_dti.y[batch_dti.y == 2] = 1 
            
            logits = model(batch_dti, batch_fmri)
            
            # Get Class 1 Probability (for AUC)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_dti.y.cpu().numpy())

    # --- 3. METRICS ---
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity Calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC Calculation
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5 # Fail-safe for single class batches

    print("\n" + "="*40)
    print("      MEDICAL DIAGNOSTIC REPORT      ")
    print("="*40)
    print(f"✅ Accuracy:    {acc*100:.2f}%  (Overall correctness)")
    print(f"🔍 Sensitivity: {rec*100:.2f}%  (Ability to detect PD)")
    print(f"🛡️ Specificity: {specificity*100:.2f}%  (Ability to confirm Healthy)")
    print(f"⚖️ F1 Score:    {f1:.3f}     (Balance of Prec/Recall)")
    print(f"📈 ROC-AUC:     {auc:.3f}     (Ranking Ability)")
    print("="*40)

    print("\n--- Confusion Matrix ---")
    print(f"True Healthy: {tn} | False Alarms (FP): {fp}")
    print(f"Missed PD (FN): {fn} | Caught PD (TP):    {tp}")

    # --- 4. VISUALIZATION: Confusion Matrix & ROC ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix Plot
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Pred Healthy', 'Pred PD'],
                yticklabels=['True Healthy', 'True PD'])
    ax1.set_title("Confusion Matrix")

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate (1 - Specificity)')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('metrics_report.png')
    print("\n✅ Saved detailed report to 'metrics_report.png'")

if __name__ == "__main__":
    evaluate_model()