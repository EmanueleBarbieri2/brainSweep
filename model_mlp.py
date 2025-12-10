import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleBrainCLIP(nn.Module):
    def __init__(self, num_nodes=90, embedding_dim=32, hidden_dim=1024, dropout=0.5):
        super().__init__()
        
        # Input size: (90 * 89) / 2 = 4005 unique edges
        self.input_dim = int(num_nodes * (num_nodes - 1) / 2)
        
        # --- fMRI Encoder ---
        self.fmri_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # --- DTI Encoder ---
        self.dti_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # CLIP Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_upper_tri(self, matrices):
        # Flattens (Batch, 90, 90) -> (Batch, 4005)
        # We only take the upper triangle to avoid redundant/diagonal info
        triu_indices = torch.triu_indices(matrices.shape[1], matrices.shape[2], offset=1)
        return matrices[:, triu_indices[0], triu_indices[1]]

    def forward(self, fmri, dti):
        # 1. Flatten
        fmri_flat = self.get_upper_tri(fmri)
        dti_flat = self.get_upper_tri(dti)
        
        # 2. Encode
        fmri_emb = self.fmri_encoder(fmri_flat)
        dti_emb = self.dti_encoder(dti_flat)
        
        # 3. Normalize (Crucial for CLIP/VICReg)
        fmri_emb = F.normalize(fmri_emb, p=2, dim=1)
        dti_emb = F.normalize(dti_emb, p=2, dim=1)
        
        # Return 3 values (Scale is needed for CLIP, ignored for VICReg)
        return fmri_emb, dti_emb, self.logit_scale.exp()