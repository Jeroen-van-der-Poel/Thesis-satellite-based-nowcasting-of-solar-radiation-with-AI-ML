import os
import h5py
import torch
from torch.utils.data import DataLoader
from RawData.netCDFDataset import NetCDFNowcastingDataset
from tqdm import tqdm

def save_netcdf_to_hdf5(netcdf_root, hdf5_path, batch_size=1):
    dataset = NetCDFNowcastingDataset(root_dir=netcdf_root)
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
        for batch in tqdm(loader, desc="Saving HDF5"):
            data = batch["vil"].numpy()
            for i in range(data.shape[0]):
                dset[idx] = data[i]
                idx += 1

    print(f"Saved {total_samples} samples to {hdf5_path}")

if __name__ == "__main__":
    raw_train = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/"
    raw_test = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/"

    output_train = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data/train_data.h5"
    output_test = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data/test_data.h5"

    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    save_netcdf_to_hdf5(raw_train, output_train)

    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    save_netcdf_to_hdf5(raw_test, output_test)