import os
import h5py
import torch
from netCDFDataset import NetCDFNowcastingDataset
import numpy as np

output_dir_train = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data'
output_dir_test = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data'
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

train_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/')
test_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/')

samples_per_file = 500

def save_batched_hdf5(dataset, output_dir, batch_size):
    file_id = 0
    sample_in_file = 0
    total_valid_samples = 0
    hf = None

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            sample_data = sample["vil"].half().numpy()  # shape: [20, H, W, 1]

            # Create new file if starting a new batch
            if sample_in_file == 0:
                if hf:
                    hf.close()
                hf_path = os.path.join(output_dir, f"{file_id:06d}.h5")
                hf = h5py.File(hf_path, 'w')

            dataset_name = f'sample_{sample_in_file:03d}'
            hf.create_dataset(dataset_name, data=sample_data.astype(np.float16), compression="gzip")
            sample_in_file += 1
            total_valid_samples += 1

            # If batch is full, reset
            if sample_in_file == batch_size:
                hf.close()
                file_id += 1
                sample_in_file = 0

        except IndexError:
            continue

    # Close last file if still open
    if hf and not hf.closed:
        hf.close()

    print(f"Saved {file_id + (1 if sample_in_file > 0 else 0)} HDF5 files to {output_dir}")
    print(f"Total valid samples saved: {total_valid_samples}")

save_batched_hdf5(train_dataset, output_dir_train, samples_per_file)
save_batched_hdf5(test_dataset, output_dir_test, samples_per_file)
