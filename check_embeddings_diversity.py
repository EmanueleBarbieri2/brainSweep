import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import your modules
from dataloader import create_dataloaders
from model import ContrastiveModel

def get_similarity_stats(embeddings):
    """
    Computes pairwise similarity statistics for a batch of embeddings.
    """
    # Matrix multiplication of normalized vectors = Cosine Similarity
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Mask out the diagonal (self-similarity is always 1.0)
    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool)
    off_diag_sim = sim_matrix[~mask].view(-1)
    
    stats = {
        'avg': off_diag_sim.mean().item(),
        'max': off_diag_sim.max().item(),
        'min': off_diag_sim.min().item(),
        'std': off_diag_sim.std().item(),
        'matrix': sim_matrix.cpu().numpy()
    }
    return stats

def check_diversity():
    # 1. Setup Data & Model
    print("Loading Data...")
    train_loader, _ = create_dataloaders(batch_size=32)
    
    # Grab a single batch
    batch_dti, batch_fmri = next(iter(train_loader))
    
    # --- FIX IS HERE ---
    # Get dimensions dynamically from the batch
    num_node_features = batch_dti.x.shape[1]
    # Calculate number of nodes per graph (Total Nodes in Batch / Batch Size)
    num_nodes = batch_dti.num_nodes // batch_dti.num_graphs
    
    print(f"Initializing Hybrid Model (Nodes: {num_nodes}, Feats: {num_node_features})...")
    
    # Pass BOTH arguments to the model
    model = ContrastiveModel(
        num_nodes=num_nodes, 
        num_node_features=num_node_features
    )
    
    model.eval() # Eval mode to stabilize stats

    # 2. Forward Pass (Untrained)
    print("Running Forward Pass...")
    with torch.no_grad():
        z_dti, z_fmri = model(batch_dti, batch_fmri)

    # 3. Compute Stats for Both
    dti_stats = get_similarity_stats(z_dti)
    fmri_stats = get_similarity_stats(z_fmri)

    # 4. Report Results
    def report(name, stats):
        print("\n" + "="*40)
        print(f"   DIVERSITY CHECK: {name}")
        print("="*40)
        print(f"Avg Similarity:  {stats['avg']:.4f}  (Target: ~0.0 to 0.5)")
        print(f"Max Similarity:  {stats['max']:.4f}  (Should be < 0.95)")
        print(f"Std Deviation:   {stats['std']:.4f}  (Higher is better)")
        
        if stats['avg'] > 0.9:
            print(f"❌ CRITICAL FAILURE: {name} Mode Collapse.")
        elif stats['avg'] > 0.7:
            print(f"⚠️ WARNING: High {name} Similarity.")
        else:
            print(f"✅ SUCCESS: Good {name} Diversity.")

    report("DTI (Structure)", dti_stats)
    report("fMRI (Function)", fmri_stats)
    print("="*40 + "\n")

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # DTI Heatmap
    sns.heatmap(dti_stats['matrix'], cmap="viridis", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title(f"DTI Similarity (Avg: {dti_stats['avg']:.2f})")
    axes[0].set_xlabel("Patient Index")
    axes[0].set_ylabel("Patient Index")

    # fMRI Heatmap
    sns.heatmap(fmri_stats['matrix'], cmap="viridis", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title(f"fMRI Similarity (Avg: {fmri_stats['avg']:.2f})")
    axes[1].set_xlabel("Patient Index")
    axes[1].set_yticks([]) 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_diversity()