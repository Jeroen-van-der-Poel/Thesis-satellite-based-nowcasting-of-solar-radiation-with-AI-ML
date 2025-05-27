import h5py
import torch
from torch.utils.data import Dataset

class HDF5NowcastingDataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.file = None  

    def _init_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            self.vil = self.file['vil']

    def __len__(self):
        self._init_file()
        print(f"Dataset length: {self.vil}")
        return self.vil.shape[0]

    def __getitem__(self, idx):
        self._init_file()
        return torch.from_numpy(self.vil[0][idx])
