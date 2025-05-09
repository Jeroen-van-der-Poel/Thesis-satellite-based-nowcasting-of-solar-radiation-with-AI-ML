import os
import h5py
import torch
from netCDFDataset import NetCDFNowcastingDataset
import numpy as np
import psutil
import gc
import time

output_dir_train = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data'
output_dir_test = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data'
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

train_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/')
test_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/')

samples_per_file = 500

def save_batched_hdf5(dataset, output_dir, batch_size):
    sample_id = 0
    batch = []
    total_valid_samples = 0
    pid = os.getpid()
    process = psutil.Process(pid)

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            vil_tensor = sample["vil"].numpy()  # (20, H, W, 1)
            batch.append(vil_tensor)
            total_valid_samples += 1

            if len(batch) == batch_size:
                array = np.stack(batch, axis=0)  # Shape: (500, 20, H, W, 1)
                file_path = os.path.join(output_dir, f"{sample_id:06d}.h5")
                try:
                    with h5py.File(file_path, 'w') as hf:
                        hf.create_dataset("vil", data=array, compression="gzip", dtype='f2')
                except Exception as e:
                    print(f"Failed to write {file_path}: {e}")
                batch = []
                sample_id += 1

                # Monitor memory and open file handles
                mem = process.memory_info().rss / (1024 ** 2)  # in MB
                print(f"[File {sample_id:06d}] Memory usage: {mem:.2f} MB")
                print(f"Open files: {len(process.open_files())}")

                # Clean up
                del array
                gc.collect()
                time.sleep(0.01)

        except IndexError:
            continue

    # Save any remaining samples
    if batch:
        array = np.stack(batch, axis=0)
        file_path = os.path.join(output_dir, f"{sample_id:06d}.h5")
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset("vil", data=array, compression="gzip", dtype='f2')
        sample_id += 1

    print(f"Saved {sample_id} HDF5 files to {output_dir}")
    print(f"Total valid samples saved: {total_valid_samples}")

save_batched_hdf5(train_dataset, output_dir_train, samples_per_file)
save_batched_hdf5(test_dataset, output_dir_test, samples_per_file)