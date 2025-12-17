import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter

class DiverseGINE(nn.Module):
    def __init__(self, num_node_features, num_nodes, hidden_dim=64, embed_dim=128, 
                 dropout=0.1, num_gnn_layers=2, projection_layers=3, activation="relu"):
        super().__init__()
        
        # Select Activation
        self.act_name = activation
        if activation == "relu": self.act = nn.ReLU()
        elif activation == "gelu": self.act = nn.GELU()
        elif activation == "tanh": self.act = nn.Tanh()
        
        # 1. Hybrid Input Encoder
        input_dim = num_node_features + num_nodes + 1 
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        self.register_buffer('eye', torch.eye(num_nodes))

        # 2. Edge Encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. Dynamic GINE Layers
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.act,
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))

        # 4. Dynamic Projection Head
        flatten_dim = (num_nodes * hidden_dim) + num_nodes
        
        head_layers = []
        # Input Layer
        head_layers.append(nn.Linear(flatten_dim, 512))
        head_layers.append(nn.LayerNorm(512))
        head_layers.append(self.act)
        head_layers.append(nn.Dropout(dropout))
        
        # Hidden Layers (Dynamic)
        curr_dim = 512
        for _ in range(projection_layers - 2): # -2 because we have input and output fixed
            head_layers.append(nn.Linear(curr_dim, curr_dim // 2))
            head_layers.append(nn.LayerNorm(curr_dim // 2))
            head_layers.append(self.act)
            head_layers.append(nn.Dropout(dropout))
            curr_dim = curr_dim // 2
            
        # Output Layer
        head_layers.append(nn.Linear(curr_dim, embed_dim))
        
        self.projection = nn.Sequential(*head_layers)

    def _sparsify_edges(self, edge_attr, keep_ratio):
        if keep_ratio >= 1.0: return edge_attr
        magnitude = torch.abs(edge_attr).flatten()
        k = int(magnitude.numel() * keep_ratio)
        if k < 1: k = 1
        threshold = torch.kthvalue(magnitude, magnitude.numel() - k + 1).values
        mask = magnitude >= threshold
        return edge_attr * mask.view(-1, 1)

    def _get_abs_node_strength(self, edge_index, edge_attr, num_nodes):
        row, col = edge_index
        if edge_attr.dim() > 1: edge_attr = edge_attr.flatten()
        abs_attr = torch.abs(edge_attr)
        out = scatter(abs_attr, col, dim=0, dim_size=num_nodes, reduce='sum')
        return out.view(-1, 1)

    def forward(self, data, keep_ratio):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if edge_attr.dim() == 1: edge_attr = edge_attr.view(-1, 1)
        sparse_attr = self._sparsify_edges(edge_attr, keep_ratio)

        # Input Injection
        strength = self._get_abs_node_strength(edge_index, sparse_attr, x.size(0))
        batch_size = batch.max().item() + 1
        identities = self.eye.repeat(batch_size, 1)[:x.size(0)]
        x = torch.cat([x, identities, strength], dim=1)
        x = self.input_encoder(x)

        # Edge Encoding
        non_zero_mask = sparse_attr != 0
        if non_zero_mask.sum() > 0:
            mean_edge = sparse_attr[non_zero_mask].mean()
            std_edge = sparse_attr[non_zero_mask].std() + 1e-6
        else:
            mean_edge = 0; std_edge = 1
        
        delta_edge = torch.zeros_like(sparse_attr)
        delta_edge[non_zero_mask] = ((sparse_attr[non_zero_mask] - mean_edge) / std_edge) * 2.0
        edge_embeddings = self.edge_encoder(delta_edge)

        # Message Passing (Dynamic Loop)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_embeddings)
            x = self.act(x)
        
        # Flatten
        flat_gnn = x.view(batch_size, -1)
        flat_strength = strength.view(batch_size, -1)
        final_vec = torch.cat([flat_gnn, flat_strength], dim=1)
        
        return self.projection(final_vec)

class ContrastiveModel(nn.Module):
    def __init__(self, num_nodes, num_node_features, embed_dim=128, hidden_dim=64, 
                 dropout=0.1, num_gnn_layers=2, projection_layers=3, activation="relu"):
        super().__init__()
        
        self.dti_encoder = DiverseGINE(num_node_features, num_nodes, hidden_dim, embed_dim, 
                                     dropout, num_gnn_layers, projection_layers, activation)
        self.fmri_encoder = DiverseGINE(num_node_features, num_nodes, hidden_dim, embed_dim, 
                                      dropout, num_gnn_layers, projection_layers, activation)
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, dti_data, fmri_data, dti_keep=0.2, fmri_keep=0.2, use_batch_centering=True):
        z_dti = self.dti_encoder(dti_data, keep_ratio=dti_keep)
        z_fmri = self.fmri_encoder(fmri_data, keep_ratio=fmri_keep)
        
        if use_batch_centering:
            z_dti = z_dti - z_dti.mean(dim=0, keepdim=True)
            z_fmri = z_fmri - z_fmri.mean(dim=0, keepdim=True)
        
        return F.normalize(z_dti, dim=1), F.normalize(z_fmri, dim=1)