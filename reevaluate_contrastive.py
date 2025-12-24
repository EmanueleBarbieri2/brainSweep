import os
os.environ["q"] = "true" 

import torch
import torch.nn as nn
from dataloader import create_dataloaders
from model import ContrastiveModel

# CONFIG (Must match the Fine-Tuned Model architecture)
CONFIG = {
    'batch_size': 16,
    'hidden_dim': 64,
    'embed_dim': 128,
    'dropout': 0.0, # No dropout for evaluation
    'num_gnn_layers': 2,
    'projection_layers': 3,
    'activation': 'relu',
    'dti_keep': 0.8,    
    'fmri_keep': 0.5,   
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_finetuned_model.pth' 
}

def validate_metrics(model, loader, device):
    model.eval()
    dti_embeds = []
    fmri_embeds = []
    
    print("Extracting embeddings for alignment check...")
    with torch.no_grad():
        for batch_dti, batch_fmri in loader:
            batch_dti = batch_dti.to(device)
            batch_fmri = batch_fmri.to(device)
            
            # Use the fine-tuning "keep" ratios to be consistent
            z_dti, z_fmri = model(
                batch_dti, batch_fmri, 
                dti_keep=CONFIG['dti_keep'], 
                fmri_keep=CONFIG['fmri_keep'],
                use_batch_centering=False
            )
            
            # Normalize vectors (Cosine Similarity requires normalized vectors)
            dti_embeds.append(torch.nn.functional.normalize(z_dti, dim=1))
            fmri_embeds.append(torch.nn.functional.normalize(z_fmri, dim=1))
            
    # Concatenate all batches
    dti_feats = torch.cat(dti_embeds)
    fmri_feats = torch.cat(fmri_embeds)
    
    # Similarity Matrix: (N_subjects x N_subjects)
    sim_matrix = dti_feats @ fmri_feats.T
    
    n = sim_matrix.size(0)
    targets = torch.arange(n).to(device).view(-1, 1)
    
    # Rank candidates
    # For each DTI, which fMRI is closest?
    _, indices = torch.sort(sim_matrix, dim=1, descending=True)
    
    # Calculate Metrics
    top1 = (indices[:, 0:1] == targets).sum().item() / n * 100
    top5 = (indices[:, :5] == targets).sum().item() / n * 100
    
    # MRR (Mean Reciprocal Rank)
    # Find where the true target is in the ranked list
    pos = torch.where(indices == targets)[1]
    mrr = (1.0 / (pos + 1).float()).mean().item()
    
    return top1, top5, mrr

def run_evaluation():
    print(f"--- Re-evaluating Contrastive Metrics on {CONFIG['model_path']} ---")
    
    # Load Validation Data
    _, val_loader = create_dataloaders(CONFIG['batch_size'], node_drop_prob=0.0)
    
    sample, _ = next(iter(val_loader))
    real_nodes = sample.num_nodes // sample.num_graphs
    real_feats = sample.x.shape[1]

    # Initialize Base Model
    model = ContrastiveModel(
        num_nodes=real_nodes, num_node_features=real_feats,
        embed_dim=CONFIG['embed_dim'], hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout'], num_gnn_layers=CONFIG['num_gnn_layers'],
        projection_layers=CONFIG['projection_layers'], activation=CONFIG['activation']
    ).to(CONFIG['device'])
    
    # LOAD WEIGHTS WITH STRICT=FALSE
    try:
        state_dict = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
        
        # This will load the Encoder weights and IGNORE the 'head' weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("✅ Weights loaded (ignoring classifier head).")
        # print(f"Ignored keys (Classifier Head): {unexpected}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Calculate
    top1, top5, mrr = validate_metrics(model, val_loader, CONFIG['device'])
    
    print("\n" + "="*40)
    print("   CONTRASTIVE METRICS (Fine-Tuned)   ")
    print("="*40)
    print(f"🥇 Top-1 Accuracy: {top1:.2f}%")
    print(f"🥈 Top-5 Accuracy: {top5:.2f}%")
    print(f"🎯 MRR Score:      {mrr:.4f}")
    print("="*40)
    
    print("\nInterpretation:")
    if top1 > 80:
        print("✅ Identity preserved! The model knows both Disease AND Individual identity.")
    elif top1 < 50:
        print("⚠️ Identity lost. The model collapsed patients into generic 'Healthy' vs 'PD' blobs.")
    else:
        print("ℹ️ Identity partially preserved. Some trade-off occurred.")

if __name__ == "__main__":
    run_evaluation()