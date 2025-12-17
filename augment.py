import torch

class GraphAugment:
    def __init__(self, node_drop_prob=0.1, edge_noise_std=0.05):
        self.node_drop_prob = node_drop_prob
        self.edge_noise_std = edge_noise_std

    def __call__(self, data):
        data = data.clone()
        
        # Node Drop
        if self.node_drop_prob > 0:
            num_nodes = data.num_nodes
            node_mask = torch.rand(num_nodes) > self.node_drop_prob
            
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                src, dst = data.edge_index
                edge_keep_mask = node_mask[src] & node_mask[dst]
                data.edge_index = data.edge_index[:, edge_keep_mask]
                if data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[edge_keep_mask]

        # Edge Noise
        if self.edge_noise_std > 0 and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            noise = torch.randn_like(data.edge_attr) * self.edge_noise_std
            data.edge_attr += noise
            
        return data