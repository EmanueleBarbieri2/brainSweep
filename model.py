import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation 
from torch_geometric.utils import dense_to_sparse
import numpy as np

class DualGNN_CLIP(nn.Module):
    def __init__(self, num_nodes=90, hidden_dim=64, out_dim=32, num_layers=1, heads=4, dropout=0.5):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # --- 1. ENCODERS ---
        # Metric Projection: Strength, Entropy -> Hidden/4
        self.graph_metric_proj = nn.Linear(2, hidden_dim // 4) 
        
        base_in_dim = num_nodes + (hidden_dim // 4)
        
        self.dti_feat_mlp = nn.Sequential(
            nn.Linear(base_in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.fmri_feat_mlp = nn.Sequential(
            nn.Linear(base_in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        
        self.dti_input_norm = nn.InstanceNorm1d(hidden_dim)
        self.fmri_input_norm = nn.InstanceNorm1d(hidden_dim)

        # --- 2. GNN LAYERS ---
        self.dti_layers = nn.ModuleList()
        self.dti_norms = nn.ModuleList()
        self.fmri_pos_layers = nn.ModuleList()
        self.fmri_neg_layers = nn.ModuleList()
        self.fmri_norms = nn.ModuleList()
        
        if num_layers > 0:
            for _ in range(num_layers):
                self.dti_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))
                self.dti_norms.append(nn.LayerNorm(hidden_dim))
                
                self.fmri_pos_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))
                self.fmri_neg_layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))
                self.fmri_norms.append(nn.LayerNorm(hidden_dim))

        # --- 3. POOLING ---
        self.dti_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        self.fmri_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))

        # --- 4. OUTPUT HEADS ---
        final_in_dim = hidden_dim * 2 if num_layers > 0 else hidden_dim
        
        self.dti_out = nn.Sequential(
            nn.Linear(final_in_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )
        self.fmri_out = nn.Sequential(
            nn.Linear(final_in_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.4)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _compute_node_features(self, matrix):
        """
        Calculates node metrics.
        FIX: Intra-Subject Standardization.
        Instead of raw values, we express metrics as deviations from the patient's own mean.
        This removes global bias (e.g., "this scan is just brighter").
        """
        # 1. Strength (Weighted Degree)
        strength = matrix.abs().sum(dim=2, keepdim=True) 
        # Z-score normalization per patient
        mean = strength.mean(dim=1, keepdim=True)
        std = strength.std(dim=1, keepdim=True) + 1e-6
        strength = (strength - mean) / std
        
        # 2. Entropy
        probs = matrix.abs() / (matrix.abs().sum(dim=2, keepdim=True) + 1e-6)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=2, keepdim=True)
        # Z-score normalization per patient
        mean_ent = entropy.mean(dim=1, keepdim=True)
        std_ent = entropy.std(dim=1, keepdim=True) + 1e-6
        entropy = (entropy - mean_ent) / std_ent
        
        return torch.cat([strength, entropy], dim=2) 

    def forward_dti(self, dti_matrix, batch_size):
        device = dti_matrix.device
        
        ###
        # DropEdge (Training Only)
        #if self.training:
            #mask = torch.rand_like(dti_matrix) > 0.2
            #dti_matrix = dti_matrix * mask
        ###

        # 1. Contrast
        dti_matrix = dti_matrix ** 3 
        
        # 2. Metrics & Features
        node_metrics = self._compute_node_features(dti_matrix)
        metric_embeds = self.graph_metric_proj(node_metrics)
        
        identity = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        augmented = (dti_matrix * 100.0) + identity
        
        # [Identity+Row (90), Metrics (16)]
        features_in = torch.cat([augmented, metric_embeds], dim=2)
        x_raw = self.dti_feat_mlp(features_in.view(-1, features_in.shape[-1]))
        
        # Norm
        x_raw = x_raw.view(batch_size, self.num_nodes, -1).transpose(1, 2)
        x_raw = self.dti_input_norm(x_raw)
        x_raw = x_raw.transpose(1, 2).reshape(-1, x_raw.shape[1])
        
        x = x_raw

        # 3. Build Graph
        edge_index_list, edge_attr_list, batch_vec_list = [], [], []
        for i in range(batch_size):
            if self.num_layers > 0:
                edge_idx, edge_att = dense_to_sparse(dti_matrix[i])
                edge_index_list.append(edge_idx + (i * self.num_nodes))
                edge_attr_list.append(edge_att)
            batch_vec_list.append(torch.full((self.num_nodes,), i, device=device))
        
        batch_vec = torch.cat(batch_vec_list, dim=0)
        
        # 4. GAT Stack
        x_gnn = x
        if self.num_layers > 0 and len(edge_index_list) > 0:
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_attr = torch.cat(edge_attr_list, dim=0)
            
            for layer, norm in zip(self.dti_layers, self.dti_norms):
                x_in = x_gnn
                x_gnn = layer(x_gnn, edge_index, edge_attr=edge_attr)
                x_gnn = norm(x_gnn)
                x_gnn = F.relu(x_gnn)
                x_gnn = F.dropout(x_gnn, p=self.dropout, training=self.training)
                x_gnn = x_gnn + x_in
        
        # 5. Pooling & Weighted Jump
        pool_gnn = self.dti_pool(x_gnn, batch_vec)
        
        if self.num_layers > 0:
            pool_raw = self.dti_pool(x_raw, batch_vec)
            
            # FIX: Amplify the raw signal (unique) over the GNN signal (smoothed)
            # Multiplying by 2.0 forces the MLP to pay attention to the "spiky" raw data
            combined = torch.cat([pool_gnn, pool_raw * 2.0], dim=1)
        else:
            combined = pool_gnn

        node_embeds = x_gnn.view(batch_size, self.num_nodes, -1)
        return self.dti_out(combined), node_embeds

    def forward_fmri(self, fmri_matrix, batch_size):
        device = fmri_matrix.device
        
        # 1. Contrast
        fmri_matrix = fmri_matrix.sign() * (fmri_matrix.abs() ** 3)
        
        # 2. Metrics
        node_metrics = self._compute_node_features(fmri_matrix)
        metric_embeds = self.graph_metric_proj(node_metrics)

        identity = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        augmented = (fmri_matrix * 100.0) + identity
        
        features_in = torch.cat([augmented, metric_embeds], dim=2)
        x_raw = self.fmri_feat_mlp(features_in.view(-1, features_in.shape[-1]))
        
        x_raw = x_raw.view(batch_size, self.num_nodes, -1).transpose(1, 2)
        x_raw = self.fmri_input_norm(x_raw)
        x_raw = x_raw.transpose(1, 2).reshape(-1, x_raw.shape[1])
        
        x = x_raw

        # 3. Build Graph
        batch_vec_list = []
        pos_idx, pos_att, neg_idx, neg_att = [], [], [], []

        for i in range(batch_size):
            if self.num_layers > 0:
                mat = fmri_matrix[i]
                pos_mask = mat > 0
                neg_mask = mat < 0 
                
                p_idx, p_att_ = dense_to_sparse(mat * pos_mask.float())
                pos_idx.append(p_idx + (i * self.num_nodes))
                pos_att.append(p_att_)
                
                n_idx, n_att_ = dense_to_sparse(mat * neg_mask.float())
                neg_idx.append(n_idx + (i * self.num_nodes))
                neg_att.append(n_att_.abs())
            
            batch_vec_list.append(torch.full((self.num_nodes,), i, device=device))

        batch_vec = torch.cat(batch_vec_list, dim=0)

        # 4. GAT
        x_gnn = x
        if self.num_layers > 0:
            pos_index = torch.cat(pos_idx, dim=1) if any(p.numel()>0 for p in pos_idx) else torch.empty((2,0), dtype=torch.long, device=device)
            pos_attr = torch.cat(pos_att, dim=0) if any(p.numel()>0 for p in pos_att) else torch.empty((0,), device=device)
            neg_index = torch.cat(neg_idx, dim=1) if any(n.numel()>0 for n in neg_idx) else torch.empty((2,0), dtype=torch.long, device=device)
            neg_attr = torch.cat(neg_att, dim=0) if any(n.numel()>0 for n in neg_att) else torch.empty((0,), device=device)

            for i in range(self.num_layers):
                x_in = x_gnn
                if pos_index.size(1) > 0:
                    x_pos = self.fmri_pos_layers[i](x_gnn, pos_index, edge_attr=pos_attr)
                else:
                    x_pos = torch.zeros_like(x_gnn)
                if neg_index.size(1) > 0:
                    x_neg = self.fmri_neg_layers[i](x_gnn, neg_index, edge_attr=neg_attr)
                else:
                    x_neg = torch.zeros_like(x_gnn)

                x_gnn = x_pos + x_neg 
                x_gnn = self.fmri_norms[i](x_gnn)
                x_gnn = F.relu(x_gnn)
                x_gnn = F.dropout(x_gnn, p=self.dropout, training=self.training)
                x_gnn = x_gnn + x_in

        # 5. Pooling
        pool_gnn = self.fmri_pool(x_gnn, batch_vec)
        
        if self.num_layers > 0:
            pool_raw = self.fmri_pool(x_raw, batch_vec)
            # Weighted Skip for fMRI too
            combined = torch.cat([pool_gnn, pool_raw * 2.0], dim=1)
        else:
            combined = pool_gnn
            
        node_embeds = x_gnn.view(batch_size, self.num_nodes, -1)
        return self.fmri_out(combined), node_embeds

    def forward(self, fmri_batch, dti_batch):
        batch_size = fmri_batch.shape[0]
        f_glob, f_nodes = self.forward_fmri(fmri_batch, batch_size)
        d_glob, d_nodes = self.forward_dti(dti_batch, batch_size)
        
        f_glob = F.normalize(f_glob, p=2, dim=1)
        d_glob = F.normalize(d_glob, p=2, dim=1)
        
        return f_glob, d_glob, self.logit_scale.exp(), f_nodes, d_nodes