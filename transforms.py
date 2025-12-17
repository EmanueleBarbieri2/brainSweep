import torch
import torch.nn.functional as F

class BrainGraphTransform:
    def __init__(self, stats_path=None):
        """
        Args:
            stats_path (str): Path to saved normalization statistics (min, max, mean, std).
        """
        self.stats = None
        if stats_path:
            try:
                self.stats = torch.load(stats_path, weights_only=False)
                print(f"Loaded normalization stats from {stats_path}")
            except FileNotFoundError:
                print("Warning: Stats file not found. Pre-computed normalization skipped.")

        # Placeholders for Coordinate Normalization
        # MNI space is roughly bounded by +/- 100mm
        self.coord_max = 100.0 

    def __call__(self, data, modality):
        """
        Applies normalization to a single Data object.
        modality: 'DTI' or 'fMRI'
        """
        # --- 1. Normalize Coordinates (Node Features) ---
        # Scale from approx [-90, 90] to [-0.9, 0.9]
        if data.x is not None:
            data.x = data.x / self.coord_max

        # --- 2. Normalize Edges (Connectivity) ---
        if self.stats is None:
            # If no stats provided, return as is (or implement on-the-fly minmax)
            return data

        # Get stats for this modality
        stats = self.stats[modality]
        
        if modality == 'DTI':
            # Min-Max Scaling to [0, 1]
            min_val = stats['min']
            max_val = stats['max']
            
            # Clamp to handle potential outliers in new data
            data.edge_attr = torch.clamp(data.edge_attr, min_val, max_val)
            
            # Scale
            data.edge_attr = (data.edge_attr - min_val) / (max_val - min_val + 1e-6)

        elif modality == 'fMRI':
            # Fisher Z Transform (already done usually, but good to ensure)
            # Then standard scaling (Gaussian) if stats exist, 
            # OR simple clamping if we want to keep raw correlations.
            
            # For this pipeline, we rely on the z-scoring inside the model (DeltaGINE),
            # so we just ensure it's not exploding. 
            # We clip extremes to [-3, 3] to stabilize training.
            data.edge_attr = torch.clamp(data.edge_attr, -1.0, 1.0)
            
            # Optional: Shift range [-1, 1] -> [0, 1] if using ReLU models, 
            # but for our Tanh/Delta model, [-1, 1] is perfect.
            pass 

        return data