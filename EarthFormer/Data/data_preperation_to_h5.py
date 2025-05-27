import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from RawData.netCDFDataset import NetCDFNowcastingDataset
from tqdm import tqdm

def save_dataset_to_hdf5(dataset, hdf5_path, compression="gzip", compression_level=4):
    sample = dataset[0]
    if isinstance(sample, dict):
        sample = sample["vil"]
    sample_shape = sample.shape  # (T, H, W, 1)
    total_samples = len(dataset)

    # Chunk shape: use sample shape to enable faster reads per sample
    chunk_shape = (1, *sample_shape)

    with h5py.File(hdf5_path, 'w') as hf:
        dset = hf.create_dataset(
            "vil",
            shape=(total_samples, *sample_shape),
            maxshape=(total_samples, *sample_shape),  # optional for future appending
            chunks=chunk_shape,  # enables gzip to compress per-sample
            dtype='float32',
            compression=compression,
            compression_opts=compression_level
        )
        for idx in tqdm(range(total_samples), desc=f"Saving to {hdf5_path}"):
            sample = dataset[idx]
            if isinstance(sample, dict):
                sample = sample["vil"]
            dset[idx] = sample.numpy().astype(np.float32)

    print(f"Saved {total_samples} samples to {hdf5_path} with gzip compression")


if __name__ == "__main__":
    raw_train_path = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/"
    raw_test_path = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/"

    output_train_h5 = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data/train_data_2.h5"
    output_val_h5 = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/val_data/val_data_2.h5"
    output_test_h5 = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data/test_data_2.h5"

    # Load full train dataset
    full_train_dataset = NetCDFNowcastingDataset(root_dir=raw_train_path)
    total_samples = len(full_train_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    # Split into train and validation datasets
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Load train dataset
    os.makedirs(os.path.dirname(output_train_h5), exist_ok=True)
    save_dataset_to_hdf5(train_dataset, output_train_h5)

    # Load validation dataset
    os.makedirs(os.path.dirname(output_val_h5), exist_ok=True)
    save_dataset_to_hdf5(val_dataset, output_val_h5)

    # Load test dataset
    test_dataset = NetCDFNowcastingDataset(root_dir=raw_test_path)
    os.makedirs(os.path.dirname(output_test_h5), exist_ok=True)
    save_dataset_to_hdf5(test_dataset, output_test_h5)
