import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import BrainCLIPDataset

# PATHS
FMRI_DIR = "./data/train/matrices/fMRI_FC"
DTI_DIR = "./data/train/matrices/DTI_SC"

def main():
    ds = BrainCLIPDataset(FMRI_DIR, DTI_DIR)
    
    # Get first 5 patients
    print(f"--- Data Inspection (N=5) ---")
    for i in range(5):
        fmri, dti, _ = ds[i]
        
        dti_max = dti.max().item()
        dti_mean = dti.mean().item()
        dti_zeros = (dti == 0).sum().item() / dti.numel()
        
        print(f"Patient {i}: Max={dti_max:.4f} | Mean={dti_mean:.4f} | Sparsity={dti_zeros*100:.1f}%")

        if dti_max > 100:
            print("  ⚠️ WARNING: Values are HUGE. Needs Log-Transform.")
        if dti_max < 0.01 and dti_max > 0:
            print("  ⚠️ WARNING: Values are TINY. Needs Scaling.")

    # Histogram of ONE DTI matrix
    _, dti, _ = ds[0]
    vals = dti.flatten().numpy()
    vals = vals[vals > 0] # Ignore zeros for plot
    
    plt.hist(vals, bins=50)
    plt.title("Distribution of Non-Zero DTI Edge Weights")
    plt.show()

if __name__ == "__main__":
    main()