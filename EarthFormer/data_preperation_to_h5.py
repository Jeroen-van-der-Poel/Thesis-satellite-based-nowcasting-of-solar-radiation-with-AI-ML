import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import h5py
import torch
from torch.utils.data import DataLoader, random_split
from RawData.netCDFDataset import NetCDFNowcastingDataset
from tqdm import tqdm

def save_dataset_to_hdf5(dataset, hdf5_path, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_samples = len(dataset)
    sample_shape = dataset[0]["vil"].shape  # (T, H, W, 1)

    with h5py.File(hdf5_path, 'w') as hf:
        dset = hf.create_dataset(
            "vil",
            shape=(total_samples, *sample_shape),
            dtype='float32',
            compression="gzip"
        )
        idx = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Saving to {os.path.basename(hdf5_path)}"):
                data = batch["vil"].numpy()
                for i in range(data.shape[0]):
                    dset[idx] = data[i]
                    idx += 1

    print(f"Saved {total_samples} samples to {hdf5_path}")


if __name__ == "__main__":
    raw_train_path = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/"
    raw_test_path = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/"

    output_train_h5 = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data/train_data.h5"
    output_val_h5 = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/val_data/val_data.h5"
    output_test_h5 = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data/test_data.h5"

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
