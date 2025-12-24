import os
os.environ["q"] = "true" # Fix for macOS numpy issue

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from dataloader import create_dataloaders
from model import ContrastiveModel  

# CONFIG (Must match your saved model exactly)
CONFIG = {
    'batch_size': 16,
    'hidden_dim': 64,             
    'embed_dim': 128,             
    'dropout': 0.1,               
    'num_gnn_layers': 2,          
    'projection_layers': 3,       
    'activation': 'relu',         
    'dti_keep': 0.5,             # Standard inference keep ratio
    'fmri_keep': 0.2,            
    'use_batch_centering': False, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_sweep_model.pth' 
}

def extract_features(loader, model, device):
    """
    Runs the frozen model to convert Brain Graphs -> Vectors (Embeddings)
    """
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_dti, batch_fmri in loader:
            batch_dti = batch_dti.to(device)
            batch_fmri = batch_fmri.to(device)
            
            # 1. Get Aligned Embeddings (Frozen)
            z_dti, z_fmri = model(
                batch_dti, 
                batch_fmri, 
                dti_keep=CONFIG['dti_keep'], 
                fmri_keep=CONFIG['fmri_keep'],
                use_batch_centering=CONFIG['use_batch_centering']
            )
            
            # 2. FUSION: Concatenate Structure + Function
            combined_features = torch.cat([z_dti, z_fmri], dim=1) 
            
            features_list.append(combined_features.cpu().numpy())
            
            # 3. Get Labels & Merge Classes
            # -----------------------------------------------
            # Convert Class 2 (SWEDD) -> Class 1 (PD)
            # 0 = Healthy, 1 = PD, 2 = PD (SWEDD)
            # Result: 0 vs 1 (Binary)
            # -----------------------------------------------
            labels = batch_dti.y.clone()
            labels[labels == 2] = 1
            
            labels_list.append(labels.cpu().numpy())

    if len(features_list) == 0:
        return np.array([]), np.array([])

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

def train_linear_probe():
    print("--- 1. Setup Linear Probe (Binary: Healthy vs PD) ---")
    
    train_loader, val_loader = create_dataloaders(CONFIG['batch_size'], node_drop_prob=0.05)
    
    sample_dti, _ = next(iter(train_loader))
    num_node_features = sample_dti.x.shape[1]
    num_nodes = sample_dti.num_nodes // sample_dti.num_graphs

    print(f"Loading weights from {CONFIG['model_path']}...")
    
    # Initialize Model
    model = ContrastiveModel(
        num_nodes=num_nodes, 
        num_node_features=num_node_features,
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout'],
        num_gnn_layers=CONFIG['num_gnn_layers'],       
        projection_layers=CONFIG['projection_layers'], 
        activation=CONFIG['activation']                
    ).to(CONFIG['device'])
    
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("✅ Pre-trained weights loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    # FREEZE ENCODER (Crucial Step)
    for param in model.parameters():
        param.requires_grad = False

    print("\n--- 2. Feature Extraction ---")
    print("Extracting training features...")
    X_train, y_train = extract_features(train_loader, model, CONFIG['device'])
    print(f"Train Data: {X_train.shape}")
    
    print("Extracting validation features...")
    X_val, y_val = extract_features(val_loader, model, CONFIG['device'])
    print(f"Val Data:   {X_val.shape}")

    print("\n--- 3. Training Logistic Regression ---")
    # C=1.0 is standard regularization
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='liblinear')
    clf.fit(X_train, y_train)
    
    print("\n--- 4. Results ---")
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average='weighted')
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy:   {val_acc*100:.2f}%")
    print(f"Val F1 Score:   {val_f1:.3f}")
    
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, val_preds, target_names=['Healthy', 'PD']))

if __name__ == "__main__":
    train_linear_probe()