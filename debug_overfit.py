import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

# Import your actual code
from dataset import BrainCLIPDataset
from model import DualGNN_CLIP
from train import contrastive_loss, calculate_accuracy

# --- SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FMRI_DIR = "./data/train/matrices/fMRI_FC" 
DTI_DIR = "./data/train/matrices/DTI_SC"

def main():
    print("--- STARTING OVERFIT TEST ---")
    print("Goal: Reach 100% Accuracy on 4 patients.")
    print("If this fails, the model cannot learn.\n")

    # 1. Load Data
    full_dataset = BrainCLIPDataset(FMRI_DIR, DTI_DIR)
    
    # 2. Take a TINY subset (First 4 patients only)
    # We cheat and use the same 4 items over and over
    subset = torch.utils.data.Subset(full_dataset, [0, 1, 2, 3])
    loader = DataLoader(subset, batch_size=4, shuffle=False)
    
    # 3. Initialize Model
    model = DualGNN_CLIP(num_nodes=90, hidden_dim=64, out_dim=32).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # High LR for fast overfitting

    losses = []
    accuracies = []

    # 4. Train loop (Short & Fast)
    print(f"Training on {len(subset)} items for 50 epochs...")
    
    for epoch in range(50):
        batch = next(iter(loader))
        fmri, dti, _ = batch
        fmri, dti = fmri.to(DEVICE), dti.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward
        fmri_emb, dti_emb, scale = model(fmri, dti)
        
        # Loss
        loss = contrastive_loss(fmri_emb, dti_emb, scale)
        loss.backward()
        optimizer.step()
        
        # Accuracy
        acc, _ = calculate_accuracy(fmri_emb, dti_emb, top_k=1)
        acc_pct = (acc / 4) * 100
        
        losses.append(loss.item())
        accuracies.append(acc_pct)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {loss.item():.4f} | Acc: {acc_pct:.0f}%")

    # 5. Result
    if accuracies[-1] == 100.0:
        print("\n✅ SUCCESS: Model allows meaningful embedding alignment.")
        print("   The architecture is valid. You can proceed to full training.")
    else:
        print("\n❌ FAILURE: Model could not even memorize 4 patients.")
        print("   Something is fundamentally wrong (Data? Normalization? Bug?).")

    # Plot
    plt.plot(accuracies)
    plt.title("Overfitting Curve (Target: Hit 100%)")
    plt.ylabel("Top-1 Accuracy")
    plt.xlabel("Epoch")
    plt.show()

if __name__ == "__main__":
    main()