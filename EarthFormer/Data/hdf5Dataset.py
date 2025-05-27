import h5py
import torch
from torch.utils.data import Dataset
import os

class HDF5NowcastingDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.vil = f['vil'][:]
        print(f"[RAM DATASET] Loaded {len(self.vil)} samples into memory")

    def __len__(self):
        return len(self.vil)

    def __getitem__(self, idx):
        return torch.from_numpy(self.vil[idx])
