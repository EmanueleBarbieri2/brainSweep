import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import your actual code
from dataset import BrainCLIPDataset
from model import DualGNN_CLIP

# --- SETTINGS ---
FMRI_DIR = "./data/train/matrices/fMRI_FC"
DTI_DIR = "./data/train/matrices/DTI_SC"
DEVICE = torch.device("cpu") # CPU is fine for visualization
BATCH_SIZE = 16

def plot_similarity(embeddings, title):
    # Normalize first
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute Similarity (Batch x Batch)
    # Range: -1 to 1
    sim_matrix = embeddings @ embeddings.T
    
    # Convert to Numpy
    sim_np = sim_matrix.detach().numpy()
    
    # Check for Collapse
    # We look at the off-diagonal elements (Patient A vs Patient B)
    mask = ~np.eye(sim_np.shape[0], dtype=bool)
    avg_sim = sim_np[mask].mean()
    std_sim = sim_np[mask].std()
    
    print(f"\n--- {title} Statistics ---")
    print(f"  Mean Similarity (off-diagonal): {avg_sim:.4f}")
    print(f"  Std  Similarity (off-diagonal): {std_sim:.4f}")
    
    if std_sim < 1e-4:
        print("  ❌ CRITICAL FAILURE: Embeddings have COLLAPSED (Identical for all patients).")
    else:
        print("  ✅ PASS: Embeddings are distinct.")

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_np, cmap="viridis", vmin=0, vmax=1)
    plt.title(f"{title}\n(If solid yellow -> COLLAPSED)")
    plt.show()

def main():
    print("Loading Data...")
    dataset = BrainCLIPDataset(FMRI_DIR, DTI_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Get one batch
    fmri, dti, _ = next(iter(loader))
    
    print("Initializing Untrained GNN...")
    # NOTE: Set depth to what you are currently testing (e.g., 2 or 3)
    model = DualGNN_CLIP(num_nodes=90, hidden_dim=64, out_dim=32, num_layers=1, heads=1).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        # Pass data through UNTRAINED model
        # We assume your model returns 5 values now (f_glob, d_glob, scale, f_nodes, d_nodes)
        f_glob, d_glob, _, _, _ = model(fmri, dti)
        
        # Visualize fMRI Embeddings
        plot_similarity(f_glob, "Untrained fMRI Embeddings")
        
        # Visualize DTI Embeddings
        plot_similarity(d_glob, "Untrained DTI Embeddings")

if __name__ == "__main__":
    main()