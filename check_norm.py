import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transforms import BrainGraphTransform

# --- 1. Load Raw Data & Stats ---
print("Loading data...")
train_dti_list = torch.load('data/train/dti_graphs.pt', weights_only=False)
train_fmri_list = torch.load('data/train/fmri_graphs.pt', weights_only=False)

# Initialize the transform
transform = BrainGraphTransform('normalization_stats.pt')

# --- 2. Pick a Patient to Audit ---
idx = 0  # You can change this index to check different patients
raw_dti = train_dti_list[idx]
raw_fmri = train_fmri_list[idx]

# Apply Normalization
norm_dti = transform(raw_dti, modality='DTI')
norm_fmri = transform(raw_fmri, modality='fMRI')

# --- 3. Print Numerical Stats ---
def print_stats(name, tensor):
    print(f"{name:<15} | Min: {tensor.min():.4f} | Max: {tensor.max():.4f} | Mean: {tensor.mean():.4f}")

print("\n--- STATS CHECK ---")
print_stats("Raw DTI", raw_dti.edge_attr)
print_stats("Norm DTI", norm_dti.edge_attr)
print("-" * 40)
print_stats("Raw fMRI", raw_fmri.edge_attr)
print_stats("Norm fMRI", norm_fmri.edge_attr)

# --- 4. Plot Histograms ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# DTI
sns.histplot(raw_dti.edge_attr.numpy(), bins=50, ax=axes[0,0], color='blue', kde=True)
axes[0,0].set_title('Raw DTI')
sns.histplot(norm_dti.edge_attr.numpy(), bins=50, ax=axes[0,1], color='cyan', kde=True)
axes[0,1].set_title('Normalized DTI (Scaled 0-1)')

# fMRI
sns.histplot(raw_fmri.edge_attr.numpy(), bins=50, ax=axes[1,0], color='red', kde=True)
axes[1,0].set_title('Raw fMRI (Correlations)')
axes[1,0].set_xlim(-1, 1) # Force view of correlation range

sns.histplot(norm_fmri.edge_attr.numpy(), bins=50, ax=axes[1,1], color='orange', kde=True)
axes[1,1].set_title('Normalized fMRI (Fisher Z - Gaussian)')

print("\n--- VISUAL CHECK ---")
print("1. Norm DTI Max should be <= 1.0.")
print("2. Norm fMRI should look like a bell curve (Gaussian), not a flat wall.")
print("Displaying plots...")
plt.tight_layout()
plt.show()