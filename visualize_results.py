import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import your modules
from dataset import BrainCLIPDataset
from augmentations import GraphAugment  # <--- Need this for TTA
from model import DualGNN_CLIP

# --- CONFIGURATION ---
MODEL_PATH = "best_model.pth"
FMRI_DIR = "./data/val/matrices/fMRI_FC"
DTI_DIR = "./data/val/matrices/DTI_SC"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HYPERPARAMS (MUST MATCH YOUR BEST SWEEP RUN) ---

HIDDEN_DIM = 128      
EMBED_DIM = 64       
GNN_DEPTH = 1        
HEADS = 4
DROPOUT = 0.5
TTA_STEPS = 5       

def load_data_and_model():
    print("Loading Validation Data...")
    # Validation dataset usually has no transform, but we apply it manually for TTA
    val_dataset = BrainCLIPDataset(FMRI_DIR, DTI_DIR)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Loading Model from {MODEL_PATH}...")
    model = DualGNN_CLIP(
        num_nodes=90, 
        hidden_dim=HIDDEN_DIM, 
        out_dim=EMBED_DIM, 
        num_layers=GNN_DEPTH, 
        heads=HEADS, 
        dropout=DROPOUT
    ).to(DEVICE)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Hint: Did you change the model architecture (e.g., added gates/metrics) since saving?")
        return None, None

    model.eval()
    return model, val_loader

def extract_embeddings_with_tta(model, loader):
    print(f"Extracting embeddings with {TTA_STEPS}x TTA...")
    
    # Augmentor for TTA (Low jitter to smooth noise)
    tta_aug = GraphAugment(mode='jitter', jitter_sigma=0.02, threshold=0.0)
    
    final_f_embs = []
    final_d_embs = []
    all_labels = []
    
    # We need to accumulate embeddings for all patients across TTA steps
    # But loader batches data. The easiest way is to run the full loader TTA times.
    
    # Store results: List of [N_patients, Dim] tensors
    tta_f_results = []
    tta_d_results = []
    
    with torch.no_grad():
        for step in range(TTA_STEPS):
            print(f"  - TTA Pass {step+1}/{TTA_STEPS}...")
            step_f = []
            step_d = []
            step_labels = []
            
            for fmri, dti, idx in loader:
                fmri, dti = fmri.to(DEVICE), dti.to(DEVICE)
                
                # Apply Augmentation (Skip on first pass if you want 1 clean view, 
                # but training usually augments all. Let's augment all for consistency.)
                if TTA_STEPS > 1:
                    fmri = tta_aug(fmri)
                    dti = tta_aug(dti)
                
                # Forward pass
                f_e, d_e, _, _, _ = model(fmri, dti)
                
                step_f.append(f_e.cpu())
                step_d.append(d_e.cpu())
                
                if step == 0: # Only save labels once
                    step_labels.extend(idx.numpy())
            
            # Concatenate batches for this TTA step
            tta_f_results.append(torch.cat(step_f, dim=0))
            tta_d_results.append(torch.cat(step_d, dim=0))
            if step == 0:
                all_labels = np.array(step_labels)

    # Average the embeddings across TTA steps
    # Stack -> (TTA, N, Dim) -> Mean -> (N, Dim)
    print("  - Averaging views...")
    final_f_embs = torch.stack(tta_f_results).mean(dim=0)
    final_d_embs = torch.stack(tta_d_results).mean(dim=0)
    
    return final_f_embs, final_d_embs, all_labels

def plot_all(f_emb, d_emb, labels):
    # Normalize for Cosine Similarity
    f_emb = F.normalize(f_emb, p=2, dim=1)
    d_emb = F.normalize(d_emb, p=2, dim=1)
    
    # --- 1. Heatmap ---
    sim_matrix = (f_emb @ d_emb.T).numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, cmap="viridis", square=True)
    plt.title("Similarity (Yellow Diagonal = Match)")
    plt.xlabel("DTI"); plt.ylabel("fMRI")
    plt.show()
    
    # --- 2. Ranks ---
    ranks = []
    num = sim_matrix.shape[0]
    for i in range(num):
        # Correct score is at [i, i]
        target = sim_matrix[i, i]
        # Rank = how many scores in row i are > target
        r = (sim_matrix[i] > target).sum()
        ranks.append(r)
    
    ranks = np.array(ranks)
    top1 = np.sum(ranks == 0) / num
    top5 = np.sum(ranks < 5) / num
    mrr = np.mean(1.0 / (ranks + 1))
    
    print(f"\n--- RESULTS ---")
    print(f"Top-1: {top1*100:.1f}%")
    print(f"Top-5: {top5*100:.1f}%")
    print(f"MRR:   {mrr:.3f}")
    
    plt.figure(figsize=(8, 4))
    plt.hist(ranks, bins=np.arange(0, 33)-0.5, color='orange', edgecolor='black')
    plt.title(f"Rank Histogram (Top-5: {top5*100:.1f}%)")
    plt.xlabel("Rank"); plt.ylabel("Count")
    plt.show()

    # --- 3. t-SNE ---
    print("Computing t-SNE...")
    combined = torch.cat([f_emb, d_emb], dim=0).numpy()
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate='auto')
    proj = tsne.fit_transform(combined)
    
    f_2d = proj[:num]
    d_2d = proj[num:]
    
    plt.figure(figsize=(10, 8))
    # Lines
    for i in range(num):
        plt.plot([f_2d[i,0], d_2d[i,0]], [f_2d[i,1], d_2d[i,1]], 'k-', alpha=0.1)
    # Points
    plt.scatter(f_2d[:,0], f_2d[:,1], c='blue', label='fMRI', alpha=0.6)
    plt.scatter(d_2d[:,0], d_2d[:,1], c='red', marker='^', label='DTI', alpha=0.6)
    plt.legend()
    plt.title("Latent Space (Lines connect pairs)")
    plt.show()

if __name__ == "__main__":
    model, loader = load_data_and_model()
    if model:
        # Use the TTA extractor!
        f_emb, d_emb, labels = extract_embeddings_with_tta(model, loader)
        plot_all(f_emb, d_emb, labels)