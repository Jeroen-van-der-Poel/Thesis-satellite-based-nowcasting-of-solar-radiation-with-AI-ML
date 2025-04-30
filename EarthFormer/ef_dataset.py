import torch
from torch.utils.data import Dataset
import numpy as np
import tensorflow as tf
import io
from tfrecord.torch.dataset import TFRecordDataset

class SolarTFRecordTorchDataset(Dataset):
    def __init__(self, tfrecord_path, index_path):
        self.description = {
            "window_cond": "float",
            "height_cond": "float",
            "width_cond": "float",
            "depth_cond": "float",
            "raw_image_cond": "byte",
            "window_targ": "float",
            "height_targ": "float",
            "width_targ": "float",
            "depth_targ": "float",
            "raw_image_targ": "byte",
            "start_date": "byte"
        }
        self.dataset = TFRecordDataset(
            data_path=tfrecord_path,
            index_path=index_path,
            description=self.description,
            shuffle_queue_size=None
        )

    def __len__(self):
        return len(self.dataset)

    def _parse_tensor(self, byte_data, dtype=tf.float32):
        # Use tf.io.parse_tensor, then convert to numpy
        tensor = tf.io.parse_tensor(byte_data, out_type=dtype)
        return tensor.numpy()

    def __getitem__(self, idx):
        record = self.dataset[idx]

        cond = self._parse_tensor(record["raw_image_cond"])
        targ = self._parse_tensor(record["raw_image_targ"])
        date = tf.io.parse_tensor(record["start_date"], out_type=tf.string).numpy()

        cond_tensor = torch.from_numpy(cond).float()
        targ_tensor = torch.from_numpy(targ).float()

        return cond_tensor, targ_tensor, date



#For testing purposes only
ds = SolarTFRecordTorchDataset(
    trecord_path='/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data/val_data.tfrecords',
    index_path='/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data/val_data.tfrecords.index'
)

cond, targ, date = ds[0]
print(cond.shape, targ.shape, date)