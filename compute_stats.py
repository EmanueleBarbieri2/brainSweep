import torch

def make_stats():
    print("Loading Training Data to compute stats...")
    # Load raw lists
    dti_list = torch.load('data/train/dti_graphs.pt', weights_only=False)
    fmri_list = torch.load('data/train/fmri_graphs.pt', weights_only=False)

    print(f"Found {len(dti_list)} DTI graphs and {len(fmri_list)} fMRI graphs.")

    # --- 1. Compute DTI Stats ---
    print("Computing DTI Stats...")
    all_dti_edges = []
    for data in dti_list:
        if data.edge_attr is not None:
            all_dti_edges.append(data.edge_attr.flatten())
    
    # Concatenate all edges into one massive tensor
    dti_tensor = torch.cat(all_dti_edges)
    
    dti_stats = {
        'min': dti_tensor.min().item(),
        'max': dti_tensor.max().item(),
        'mean': dti_tensor.mean().item(),
        'std': dti_tensor.std().item()
    }
    print(f"  DTI Range: {dti_stats['min']:.4f} to {dti_stats['max']:.4f}")

    # --- 2. Compute fMRI Stats ---
    print("Computing fMRI Stats...")
    all_fmri_edges = []
    for data in fmri_list:
        if data.edge_attr is not None:
            all_fmri_edges.append(data.edge_attr.flatten())
            
    fmri_tensor = torch.cat(all_fmri_edges)
    
    fmri_stats = {
        'min': fmri_tensor.min().item(),
        'max': fmri_tensor.max().item(),
        'mean': fmri_tensor.mean().item(),
        'std': fmri_tensor.std().item()
    }
    print(f"  fMRI Range: {fmri_stats['min']:.4f} to {fmri_stats['max']:.4f}")

    # --- 3. Save ---
    final_stats = {
        'DTI': dti_stats,    # Capitalized keys as expected by transforms.py
        'fMRI': fmri_stats
    }
    
    torch.save(final_stats, 'normalization_stats.pt')
    print("\n✅ Saved correct 'normalization_stats.pt'. You are ready to train.")

if __name__ == "__main__":
    make_stats()