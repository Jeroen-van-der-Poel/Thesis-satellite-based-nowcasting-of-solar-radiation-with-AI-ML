import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Subset, random_split
from Data.hdf5Dataset import HDF5NowcastingDataset

class H5LightningDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, batch_size=8, num_workers=8):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_train_samples = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            print(f"Loading training dataset from {self.train_path}...")
            self.train_dataset = HDF5NowcastingDataset(self.train_path)
            print(f"Loaded {len(self.train_dataset)} samples for training.")
            self.num_train_samples = len(self.train_dataset)

        if self.val_dataset is None:
            print(f"Loading validation dataset from {self.val_path}...")
            self.val_dataset = HDF5NowcastingDataset(self.val_path)
            print(f"Loaded {len(self.val_dataset)} samples for validation.")

        if self.test_dataset is None:
            print(f"Loading testing dataset from {self.test_path}...")
            self.test_dataset = HDF5NowcastingDataset(self.test_path)
            print(f"Loaded {len(self.test_dataset)} samples for testing.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=False, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=False, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=False, pin_memory=False)