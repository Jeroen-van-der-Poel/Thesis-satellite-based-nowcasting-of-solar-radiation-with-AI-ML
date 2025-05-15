import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Subset, random_split
from RawData.netCDFDataset import NetCDFNowcastingDataset

class NetCDFLightningDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, batch_size=8, num_workers=16):
        super().__init__()
        self.train_path = train_path
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
            self.train_dataset = NetCDFNowcastingDataset(root_dir=self.train_path)
            print(f"Loaded {len(self.train_dataset)} samples for training/validation.")

            print(f"Loading validation dataset from...")
            val_split = int(0.2 * len(self.train_dataset))
            train_split = len(self.train_dataset) - val_split
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [train_split, val_split], generator=torch.Generator().manual_seed(42))
            print(f"Split into {len(self.train_dataset)} training and {len(self.val_dataset)} validation samples.")
            self.num_train_samples = len(self.train_dataset)

        if self.test_dataset is None:
            print(f"Loading testing dataset from {self.test_path}...")
            self.test_dataset = NetCDFNowcastingDataset(root_dir=self.test_path)
            print(f"Loaded {len(self.test_dataset)} samples for testing.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)