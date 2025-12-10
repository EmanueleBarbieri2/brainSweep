import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import glob

class BrainCLIPDataset(Dataset):
    def __init__(self, fmri_dir, dti_dir, fmri_transform=None, dti_transform=None):
        """
        Args:
            fmri_dir (str): Path to folder with fMRI .csv files
            dti_dir (str): Path to folder with DTI .csv files
        """
        self.fmri_transform = fmri_transform
        self.dti_transform = dti_transform
        
        # 1. Find all files
        fmri_files = glob.glob(os.path.join(fmri_dir, "*_fMRI.csv"))
        dti_files = glob.glob(os.path.join(dti_dir, "*_DTI.csv"))
        
        # 2. Extract IDs to ensure matching
        # Logic: "Patient_001_fMRI.csv" -> ID: "Patient_001"
        fmri_dict = {}
        for path in fmri_files:
            filename = os.path.basename(path)
            patient_id = filename.split('_fMRI')[0] 
            fmri_dict[patient_id] = path

        dti_dict = {}
        for path in dti_files:
            filename = os.path.basename(path)
            patient_id = filename.split('_DTI')[0]
            dti_dict[patient_id] = path

        # 3. Find Intersection (Common IDs)
        common_ids = sorted(list(set(fmri_dict.keys()) & set(dti_dict.keys())))
        
        if len(common_ids) == 0:
            raise ValueError(f"No matching pairs found! Check your filenames.\n"
                             f"Found {len(fmri_files)} fMRI files and {len(dti_files)} DTI files.")
        
        # 4. Store paths AND IDs (This fixes your error)
        self.fmri_paths = [fmri_dict[pid] for pid in common_ids]
        self.dti_paths = [dti_dict[pid] for pid in common_ids]
        self.patient_ids = common_ids 

        print(f"Dataset Successfully Loaded.")
        print(f"  - Found {len(fmri_files)} fMRI and {len(dti_files)} DTI files.")
        print(f"  - Established {len(self.fmri_paths)} valid pairs.")

    def load_csv_matrix(self, path):
        try:
            # Try loading without header
            df = pd.read_csv(path, header=None)
            # Check if first row/col are strings (headers/indices)
            if isinstance(df.iloc[0,0], str):
                df = pd.read_csv(path, index_col=0)
            # Ensure strictly numeric
            matrix = df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            return matrix
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((90, 90))

    def __len__(self):
        return len(self.fmri_paths)

    def __getitem__(self, idx):
        # Load
        fmri = self.load_csv_matrix(self.fmri_paths[idx])
        dti = self.load_csv_matrix(self.dti_paths[idx])

        # To Tensor
        fmri = torch.from_numpy(fmri).float()
        dti = torch.from_numpy(dti).float()

        # Augment
        if self.fmri_transform:
            fmri = self.fmri_transform(fmri)
        if self.dti_transform:
            dti = self.dti_transform(dti)
            
        # Nan Safety
        fmri = torch.nan_to_num(fmri, nan=0.0)
        dti = torch.nan_to_num(dti, nan=0.0)

        # Return ID for debugging
        return fmri, dti, idx