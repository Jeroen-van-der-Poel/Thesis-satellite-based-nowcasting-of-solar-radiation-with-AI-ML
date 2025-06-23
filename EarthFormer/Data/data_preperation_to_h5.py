import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import h5py
import numpy as np
from torch.utils.data import random_split
from RawData.netCDFDataset import NetCDFNowcastingDataset
from tqdm import tqdm

def save_dataset_to_hdf5(dataset, hdf5_path_norm, hdf5_path_cs, compression="gzip", compression_level=4):
    norm_sample, cs_sample = dataset[0]
    if isinstance(norm_sample, dict):
        norm_sample = norm_sample["vil"]
        cs_sample = cs_sample["vil"]
    sample_shape = norm_sample.shape 
    total_samples = len(dataset)
    chunk_shape = (1, *sample_shape)

    with h5py.File(hdf5_path_norm, 'w') as hf_norm, h5py.File(hdf5_path_cs, 'w') as hf_cs:
        dset_norm = hf_norm.create_dataset(
            "vil",
            shape=(total_samples, *sample_shape),
            maxshape=(total_samples, *sample_shape),
            chunks=chunk_shape,
            dtype='float32',
            compression=compression,
            compression_opts=compression_level
        )
        dset_cs = hf_cs.create_dataset(
            "vil",
            shape=(total_samples, *sample_shape),
            maxshape=(total_samples, *sample_shape),
            chunks=chunk_shape,
            dtype='float32',
            compression=compression,
            compression_opts=compression_level
        )

        for idx in tqdm(range(total_samples), desc=f"Saving to {hdf5_path_norm} and {hdf5_path_cs}"):
            norm_tensor, cs_tensor = dataset[idx]
            if isinstance(norm_tensor, dict):
                norm_tensor = norm_tensor["vil"]
                cs_tensor = cs_tensor["vil"]
            dset_norm[idx] = norm_tensor.numpy().astype(np.float32)
            dset_cs[idx] = cs_tensor.numpy().astype(np.float32)

        print(f"Saved {total_samples} samples to:\n- {hdf5_path_norm} (normalized)\n- {hdf5_path_cs} (sds_cs)")


if __name__ == "__main__":
    raw_train_path = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/"
    raw_test_path = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/"

    output_train_norm = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data/train_data.h5"
    output_train_cs = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data/train_data_cs.h5"

    output_val_norm = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/val_data/val_data.h5"
    output_val_cs = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/val_data/val_data_cs.h5"

    output_test_norm = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data/test_data_3.h5"
    output_test_cs = "/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data/test_data_cs_3.h5"

    # Load full train dataset
    # full_train_dataset = NetCDFNowcastingDataset(root_dir=raw_train_path)
    # total_samples = len(full_train_dataset)
    # train_size = int(0.8 * total_samples)
    # val_size = total_samples - train_size

    # # Split into train and validation datasets
    # train_dataset, val_dataset = random_split(
    #     full_train_dataset,
    #     [train_size, val_size],
    #     generator=torch.Generator().manual_seed(42)
    # )

    # # Load train dataset
    # os.makedirs(os.path.dirname(output_train_norm), exist_ok=True)
    # os.makedirs(os.path.dirname(output_train_cs), exist_ok=True)
    # save_dataset_to_hdf5(train_dataset, output_train_norm, output_train_cs)

    # # Load validation dataset
    # os.makedirs(os.path.dirname(output_val_norm), exist_ok=True)
    # os.makedirs(os.path.dirname(output_val_cs), exist_ok=True)
    # save_dataset_to_hdf5(val_dataset, output_val_norm, output_val_cs)

    # Load test dataset
    test_dataset = NetCDFNowcastingDataset(root_dir=raw_test_path)
    os.makedirs(os.path.dirname(output_test_norm), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_cs), exist_ok=True)
    save_dataset_to_hdf5(test_dataset, output_test_norm, output_test_cs)

