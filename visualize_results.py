import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from dataloader import create_dataloaders

from model import ContrastiveModel

# CONFIG: Must match the 'train_final.py' settings exactly
CONFIG = {
    'batch_size': 32,
    'hidden_dim': 64,
    'embed_dim': 128,
    'dropout': 0.1,
    'dti_keep_ratio': 0.5,
    'fmri_keep_ratio': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_sweep_model.pth' 
}

def visualize():
    print("--- Loading Best Model & Data ---")
    
    # 1. Load Data
    _, val_loader = create_dataloaders(CONFIG['batch_size'])
    
    sample_dti, _ = next(iter(val_loader))
    num_node_features = sample_dti.x.shape[1]
    num_nodes = sample_dti.num_nodes // sample_dti.num_graphs

    # 2. Initialize Model
    model = ContrastiveModel(
        num_nodes=num_nodes, 
        num_node_features=num_node_features, 
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    # 3. Load Weights
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("Tip: Make sure you are importing the correct 'ContrastiveModel' class (model_jitter.py)")
        return

    model.eval()
    
    # 4. Generate Embeddings
    dti_list = []
    fmri_list = []
    
    with torch.no_grad():
        for batch_dti, batch_fmri in val_loader:
            batch_dti = batch_dti.to(CONFIG['device'])
            batch_fmri = batch_fmri.to(CONFIG['device'])
            
            # Forward pass
            z_dti, z_fmri = model(
                batch_dti, 
                batch_fmri, 
                dti_keep=CONFIG['dti_keep_ratio'], 
                fmri_keep=CONFIG['fmri_keep_ratio']
            )
            
            # Normalize for cosine similarity
            z_dti = F.normalize(z_dti, dim=1)
            z_fmri = F.normalize(z_fmri, dim=1)
            
            dti_list.append(z_dti.cpu())
            fmri_list.append(z_fmri.cpu())

    dti_all = torch.cat(dti_list).numpy()
    fmri_all = torch.cat(fmri_list).numpy()
    
    # 5. Metrics
    print("\n--- Advanced Metrics ---")
    sim_matrix = dti_all @ fmri_all.T 
    N = sim_matrix.shape[0]
    
    pos_sims = np.diag(sim_matrix)
    mask = ~np.eye(N, dtype=bool)
    neg_sims = sim_matrix[mask]
    
    avg_pos = np.mean(pos_sims)
    avg_neg = np.mean(neg_sims)
    
    print(f"Avg Positive Similarity (Signal):  {avg_pos:.4f}")
    print(f"Avg Negative Similarity (Noise):   {avg_neg:.4f}")
    print(f"Margin:                            {avg_pos - avg_neg:.4f}")

    # 6. t-SNE Plot
    print("\n--- Running t-SNE ---")
    # Concatenate for joint t-SNE
    combined = np.concatenate([dti_all, fmri_all], axis=0) 
    
    tsne = TSNE(n_components=2, perplexity=min(30, N-1), random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(combined)
    
    dti_2d = reduced[:N]
    fmri_2d = reduced[N:]
    
    plt.figure(figsize=(10, 8))
    
    # A. Draw Lines (Faint Gray) - Shows the connection between same patient
    # If the model is good, these lines should be short.
    for i in range(N):
        plt.plot([dti_2d[i, 0], fmri_2d[i, 0]], [dti_2d[i, 1], fmri_2d[i, 1]], 
                 color='gray', alpha=0.15, linewidth=0.8, zorder=1)

    # B. Draw DTI (Blue Circles)
    plt.scatter(dti_2d[:, 0], dti_2d[:, 1], 
                color='royalblue', label='DTI (Structure)', s=60, edgecolors='white', linewidth=0.5, zorder=2)
    
    # C. Draw fMRI (Red Triangles)
    plt.scatter(fmri_2d[:, 0], fmri_2d[:, 1], 
                color='crimson', marker='^', label='fMRI (Function)', s=60, edgecolors='white', linewidth=0.5, zorder=2)

    plt.title(f"Joint Embedding Space\nSignal: {avg_pos:.2f} | Noise: {avg_neg:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('tsne_final.png', dpi=300)
    print("✅ Plot saved to 'tsne_final.png'")

if __name__ == "__main__":
    visualize()