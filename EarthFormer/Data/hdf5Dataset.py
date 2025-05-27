import h5py
import torch
from torch.utils.data import Dataset
import os

class HDF5NowcastingDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._file = None
        self._vil = None

    def _lazy_init(self):
        if self._vil is None:
            self._file = h5py.File(self.h5_path, 'r')
            self._vil = self._file['vil']  # Access remains on disk

    def __getitem__(self, idx):
        self._lazy_init()
        return torch.from_numpy(self._vil[idx])  # Still lazy-loaded

    def __len__(self):
        self._lazy_init()
        return self._vil.shape[0]
