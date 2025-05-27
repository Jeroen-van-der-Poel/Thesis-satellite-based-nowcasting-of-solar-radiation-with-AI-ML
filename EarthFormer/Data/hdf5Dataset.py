import h5py
import torch
from torch.utils.data import Dataset
import os

class HDF5NowcastingDataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self._vil = None 
        self._ensure_init()

    def _ensure_init(self):
        if self._vil is None:
            print(f"[HDF5 INIT] Loading file on worker {os.getpid()}")
            self._file = h5py.File(self.h5_path, 'r')
            self._vil = self._file['vil']

    def __len__(self):
        return self._vil.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self._vil[idx])
