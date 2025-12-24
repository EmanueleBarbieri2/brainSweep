import os
os.environ["q"] = "true" 

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from dataloader import create_dataloaders
from model import ContrastiveModel

# CONFIG
CONFIG = {
    'batch_size': 16,
    'hidden_dim': 64,
    'embed_dim': 128,
    'dropout': 0.1,
    'num_gnn_layers': 2,
    'projection_layers': 3,
    'activation': 'relu',
    'dti_keep': 0.8,    
    'fmri_keep': 0.5,   
    'num_classes': 2,   
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'best_finetuned_model.pth' 
}

# Wrapper for loading weights
class FineTuneModel(nn.Module):
    def __init__(self, pre_trained_model, num_classes):
        super().__init__()
        self.encoder = pre_trained_model
        input_dim = CONFIG['embed_dim'] * 2
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, dti, fmri):
        z_dti, z_fmri = self.encoder(
            dti, fmri, 
            dti_keep=CONFIG['dti_keep'], 
            fmri_keep=CONFIG['fmri_keep'],
            use_batch_centering=False 
        )
        return torch.cat([z_dti, z_fmri], dim=1)

def visualize_complete_tsne():
    print("--- 1. Setup ---")
    _, val_loader = create_dataloaders(CONFIG['batch_size'], node_drop_prob=0.0)
    sample_dti, _ = next(iter(val_loader))
    real_node_feats = sample_dti.x.shape[1]
    real_nodes = sample_dti.num_nodes // sample_dti.num_graphs

    base_model = ContrastiveModel(
        num_nodes=real_nodes, num_node_features=real_node_feats,
        embed_dim=CONFIG['embed_dim'], hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout'], num_gnn_layers=CONFIG['num_gnn_layers'],
        projection_layers=CONFIG['projection_layers'], activation=CONFIG['activation']
    ).to(CONFIG['device'])
    
    model = FineTuneModel(base_model, CONFIG['num_classes']).to(CONFIG['device'])
    
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        print("✅ Fine-Tuned Weights Loaded.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    model.eval()

    # Lists for plotting
    dti_embeds = []
    fmri_embeds = []
    fused_embeds = []
    labels = []
    model_predictions = []

    print("--- 2. Extracting Embeddings ---")
    with torch.no_grad():
        for batch_dti, batch_fmri in val_loader:
            batch_dti = batch_dti.to(CONFIG['device'])
            batch_fmri = batch_fmri.to(CONFIG['device'])
            
            # Binary Labels
            batch_dti.y[batch_dti.y == 2] = 1 
            
            # Get Separate Embeddings
            z_dti, z_fmri = model.encoder(
                batch_dti, batch_fmri, 
                dti_keep=CONFIG['dti_keep'], 
                fmri_keep=CONFIG['fmri_keep'],
                use_batch_centering=False
            )
            
            # Get Fused Embeddings
            z_fused = torch.cat([z_dti, z_fmri], dim=1)
            
            # Get Predictions (for background coloring)
            logits = model.head(z_fused)
            preds = logits.argmax(dim=1)

            dti_embeds.append(z_dti.cpu().numpy())
            fmri_embeds.append(z_fmri.cpu().numpy())
            fused_embeds.append(z_fused.cpu().numpy())
            labels.append(batch_dti.y.cpu().numpy())
            model_predictions.append(preds.cpu().numpy())

    # Convert to Arrays
    X_dti = np.concatenate(dti_embeds)
    X_fmri = np.concatenate(fmri_embeds)
    X_fused = np.concatenate(fused_embeds)
    y = np.concatenate(labels)
    y_pred = np.concatenate(model_predictions)
    N = len(y)

    # --- PLOT 1 PREP: ALIGNMENT (Joint t-SNE) ---
    print("--- 3. Running t-SNE (Alignment) ---")
    X_combined = np.concatenate([X_dti, X_fmri], axis=0)
    # Perplexity must be < N. Safe value: min(30, N-1)
    perp = min(30, N-1) if N > 1 else 1
    tsne_align = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_combined_2d = tsne_align.fit_transform(X_combined)
    
    dti_2d = X_combined_2d[:N]
    fmri_2d = X_combined_2d[N:]

    # --- PLOT 2 PREP: DECISION BOUNDARY (Fused t-SNE) ---
    print("--- 4. Running t-SNE (Classification) ---")
    tsne_dec = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_fused_2d = tsne_dec.fit_transform(X_fused)

    # VISUALIZATION TRICK:
    # Since t-SNE has no 'inverse_transform', we cannot project a grid back to the model.
    # Instead, we train a simple KNN on the 2D t-SNE points to approximate 
    # the "regions" that correspond to the model's Healthy vs PD predictions.
    print("--- 5. approximating Decision Boundary with KNN ---")
    background_model = KNeighborsClassifier(n_neighbors=min(5, N))
    background_model.fit(X_fused_2d, y_pred) # Train on t-SNE coords -> Model Predictions

    # Calculate Boundary Mesh
    x_min, x_max = X_fused_2d[:, 0].min() - 1, X_fused_2d[:, 0].max() + 1
    y_min, y_max = X_fused_2d[:, 1].min() - 1, X_fused_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    
    # Predict background
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = background_model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # === PLOT 1: ALIGNMENT (Linked Lines) ===
    colors = ['blue' if label == 0 else 'red' for label in y]
    
    for i in range(N):
        ax1.plot([dti_2d[i, 0], fmri_2d[i, 0]], [dti_2d[i, 1], fmri_2d[i, 1]], 
                 color=colors[i], alpha=0.3, linewidth=1)
        ax1.scatter(dti_2d[i, 0], dti_2d[i, 1], color=colors[i], marker='o', s=40, alpha=0.6)
        ax1.scatter(fmri_2d[i, 0], fmri_2d[i, 1], color=colors[i], marker='^', s=40, alpha=0.6)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2)]
    ax1.legend(custom_lines, ['Healthy (Linked)', 'PD (Linked)'])
    ax1.set_title("1. Alignment View (t-SNE)\n(Do DTI & fMRI stay close?)")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.grid(True, alpha=0.2)

    # === PLOT 2: DECISION BOUNDARY (Fused) ===
    # Background shows "Model Predicted Regions" (Approximated by KNN)
    ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # Dots show GROUND TRUTH
    scatter = ax2.scatter(X_fused_2d[:, 0], X_fused_2d[:, 1], c=y, 
                          cmap=plt.cm.coolwarm, edgecolors='k', s=80)
    
    handles, _ = scatter.legend_elements()
    ax2.legend(handles, ['Healthy', 'PD'], title="Diagnosis (Truth)")
    ax2.set_title("2. Classification View (t-SNE)\n(Background = Model Prediction Regions)")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig('complete_visualization_tsne.png', dpi=300)
    print("✅ Saved 'complete_visualization_tsne.png'")

if __name__ == "__main__":
    visualize_complete_tsne()