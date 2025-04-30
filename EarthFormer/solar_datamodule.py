from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ed_dataset import SolarTFRecordTorchDataset

class SolarLightningDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, batch_size):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SolarTFRecordTorchDataset(
            tfrecord_path=f"{self.train_path}.tfrecords",
            index_path=f"{self.train_path}.tfrecords.index"
        )
        self.val_dataset = SolarTFRecordTorchDataset(
            tfrecord_path=f"{self.val_path}.tfrecords",
            index_path=f"{self.val_path}.tfrecords.index"
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
