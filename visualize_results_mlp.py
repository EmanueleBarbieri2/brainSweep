import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import modules
from dataset import BrainCLIPDataset
from model_mlp import SimpleBrainCLIP

# --- CONFIG ---
MODEL_PATH = "best_mlp.pth" 
FMRI_DIR = "./data/val/matrices/fMRI_FC"
DTI_DIR = "./data/val/matrices/DTI_SC"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP Params (Must match training!)
HIDDEN_DIM = 2048
EMBED_DIM = 32

def load_stuff():
    print("Loading Data & MLP Model...")
    ds = BrainCLIPDataset(FMRI_DIR, DTI_DIR)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    model = SimpleBrainCLIP(num_nodes=90, hidden_dim=HIDDEN_DIM, embedding_dim=EMBED_DIM).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Weights loaded.")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None, None
        
    return model, loader

def get_embeddings(model, loader):
    model.eval()
    f_list, d_list, ids = [], [], []
    with torch.no_grad():
        for fmri, dti, idx in loader:
            fmri, dti = fmri.to(DEVICE), dti.to(DEVICE)
            f_e, d_e, _ = model(fmri, dti)
            f_list.append(f_e.cpu()); d_list.append(d_e.cpu())
            ids.extend(idx.numpy())
    return torch.cat(f_list), torch.cat(d_list), np.array(ids)

def plot_heatmap(f_emb, d_emb):
    sim = (f_emb @ d_emb.T).numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim, cmap='viridis', square=True)
    plt.title("MLP Similarity Matrix\n(Yellow Diagonal = Good)")
    plt.xlabel("DTI Patient"); plt.ylabel("fMRI Patient")
    plt.show()

def plot_ranks(f_emb, d_emb):
    sim = f_emb @ d_emb.T
    ranks = []
    for i in range(len(sim)):
        # Rank of correct match (i) in row i
        r = (sim[i] > sim[i, i]).sum().item()
        ranks.append(r)
    
    top1 = np.sum(np.array(ranks) == 0) / len(ranks)
    top5 = np.sum(np.array(ranks) < 5) / len(ranks)
    print(f"\n--- MLP RESULTS ---")
    print(f"Top-1: {top1*100:.1f}%")
    print(f"Top-5: {top5*100:.1f}%")
    
    plt.figure(figsize=(8, 5))
    plt.hist(ranks, bins=np.arange(0, 33)-0.5, color='orange', edgecolor='black')
    plt.title(f"MLP Retrieval Ranks (Top-5: {top5*100:.1f}%)")
    plt.xlabel("Rank (0 is best)"); plt.ylabel("Count")
    plt.show()

def plot_embedding_space(f_emb, d_emb, labels):
    print("Computing t-SNE...")
    # Stack features for shared projection
    combined = torch.cat([f_emb, d_emb], dim=0).numpy()
    
    # Run t-SNE (Perplexity low because N=32 is small)
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate='auto')
    proj = tsne.fit_transform(combined)
    
    # Split back
    num = f_emb.shape[0]
    f_2d = proj[:num]
    d_2d = proj[num:]
    
    plt.figure(figsize=(10, 8))
    
    # Draw connecting lines first (so they are behind points)
    for i in range(num):
        plt.plot([f_2d[i,0], d_2d[i,0]], [f_2d[i,1], d_2d[i,1]], 'k-', alpha=0.1)
    
    # Plot points
    plt.scatter(f_2d[:,0], f_2d[:,1], c='blue', marker='o', s=100, label='fMRI', alpha=0.6)
    plt.scatter(d_2d[:,0], d_2d[:,1], c='red', marker='^', s=100, label='DTI', alpha=0.6)
    
    # Add labels to a few points to verify ID matching
    for i in range(min(10, num)): # Label first 10 for clarity
        plt.text(f_2d[i,0], f_2d[i,1], str(labels[i]), color='blue', fontsize=8)
        plt.text(d_2d[i,0], d_2d[i,1], str(labels[i]), color='red', fontsize=8)

    plt.title("Latent Space (Lines connect same patient)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model, loader = load_stuff()
    if model:
        f_emb, d_emb, labels = get_embeddings(model, loader)
        
        plot_heatmap(f_emb, d_emb)
        plot_ranks(f_emb, d_emb)
        plot_embedding_space(f_emb, d_emb, labels)