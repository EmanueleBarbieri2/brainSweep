import torch
from torch.utils.data import Dataset

class PairedBrainDataset(Dataset):
    def __init__(self, dti_path, fmri_path, transform=None, augment=None):
        """
        Args:
            dti_path (str): Path to the .pt file containing DTI graphs.
            fmri_path (str): Path to the .pt file containing fMRI graphs.
            transform (callable, optional): Normalization function (BrainGraphTransform).
            augment (callable, optional): Augmentation function (GraphAugment). 
                                          Should only be provided for the Training set.
        """
        # Load the lists of Data objects
        # We use weights_only=False because these are complex PyG Data objects
        self.dti_list = torch.load(dti_path, weights_only=False)
        self.fmri_list = torch.load(fmri_path, weights_only=False)
        
        self.transform = transform
        self.augment = augment

        # Integrity Check
        if len(self.dti_list) != len(self.fmri_list):
            raise ValueError(f"Data Mismatch! DTI: {len(self.dti_list)}, fMRI: {len(self.fmri_list)}")

    def __len__(self):
        return len(self.dti_list)

    def __getitem__(self, idx):
        # 1. Get the raw graph pair
        dti_data = self.dti_list[idx]
        fmri_data = self.fmri_list[idx]

        # 2. Apply Normalization (Always)
        # We normalize first so the data is in the expected range [0, 1] or Gaussian
        if self.transform:
            dti_data = self.transform(dti_data, modality='DTI')
            fmri_data = self.transform(fmri_data, modality='fMRI')

        # 3. Apply Augmentation (Conditional)
        # Typically only for the training set. We augment DTI and fMRI independently
        # to create "challenging" positive pairs for the model.
        if self.augment:
            dti_data = self.augment(dti_data)
            fmri_data = self.augment(fmri_data)

        return dti_data, fmri_data