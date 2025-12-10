import torch
import numpy as np

class GraphAugment:
    def __init__(self, mode='mix', drop_prob=0.2, jitter_sigma=0.05, threshold=0.0):
        """
        threshold (float): If > 0, keeps only the top 'threshold' percent of edges.
                           e.g., 0.2 means "keep top 20% strongest edges".
        """
        self.mode = mode
        self.drop_prob = drop_prob
        self.jitter_sigma = jitter_sigma
        self.threshold = threshold

    def sparsify_graph(self, matrix):
        # Keeps only the top k% strongest connections (absolute value)
        if self.threshold <= 0:
            return matrix
            
        # Calculate the cutoff value for top k%
        # Flatten, sort, and find the value at the (1-threshold) percentile
        k = int(matrix.numel() * self.threshold)
        if k == 0: return matrix # Safety
        
        # Get top k values
        top_k_values, _ = torch.topk(matrix.abs().flatten(), k)
        cutoff = top_k_values[-1]
        
        # Zero out anything below cutoff
        mask = (matrix.abs() >= cutoff).float()
        return matrix * mask

    def drop_nodes(self, matrix):
        num_nodes = matrix.shape[0]
        keep_mask = torch.bernoulli(torch.ones(num_nodes) * (1 - self.drop_prob))
        return matrix * keep_mask.unsqueeze(1) * keep_mask.unsqueeze(0)

    def mask_edges(self, matrix):
        mask = torch.bernoulli(torch.ones_like(matrix) * (1 - self.drop_prob))
        return matrix * mask

    def jitter_weights(self, matrix):
        noise = torch.randn_like(matrix) * self.jitter_sigma
        return matrix + noise

    def __call__(self, matrix):
        # 1. Handle NaNs
        if torch.isnan(matrix).any():
            matrix = torch.nan_to_num(matrix)
            
        # 2. CRITICAL: Apply Thresholding (Sparsification) FIRST
        # This fixes the "Oversmoothing" problem by removing noise edges
        if self.threshold > 0:
            matrix = self.sparsify_graph(matrix)

        # 3. Apply Augmentations
        if self.mode == 'drop_nodes':
            return self.drop_nodes(matrix)
        elif self.mode == 'mask_edges':
            return self.mask_edges(matrix)
        elif self.mode == 'jitter':
            return self.jitter_weights(matrix)
        elif self.mode == 'mix':
            choice = np.random.choice(['drop', 'mask', 'jitter'])
            if choice == 'drop':
                return self.drop_nodes(matrix)
            elif choice == 'mask':
                return self.mask_edges(matrix)
            else:
                return self.jitter_weights(matrix)
        return matrix