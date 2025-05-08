import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Subset, random_split
from Data.netCDFDataset import NetCDFNowcastingDataset

class NetCDFLightningDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, batch_size=8, num_workers=4):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            train_dataset = NetCDFNowcastingDataset(root_dir=self.train_path)

            # Filter valid indices using __getitem__
            valid_indices = []
            for idx in range(len(train_dataset)):
                try:
                    _ = train_dataset[idx]  # Triggers the dark/missing checks
                    valid_indices.append(idx)
                except IndexError:
                    continue
            print(f"Found {len(valid_indices)} valid samples for training/validation.")

            valid_train_dataset = Subset(train_dataset, valid_indices)
            val_split = int(0.2 * len(valid_train_dataset))
            train_split = len(valid_train_dataset) - val_split
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(valid_train_dataset, [train_split, val_split], generator=torch.Generator().manual_seed(42))
        if self.test_dataset is None:
            self.test_dataset = NetCDFNowcastingDataset(root_dir=self.test_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
