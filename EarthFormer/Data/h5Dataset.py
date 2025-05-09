import os
import h5py
import torch
from torch.utils.data import Dataset

class PreprocessedHDF5Dataset(Dataset):
    def __init__(self, h5_dir):
        self.file_paths = sorted([
            os.path.join(h5_dir, fname)
            for fname in os.listdir(h5_dir)
            if fname.endswith(".h5")
        ])
        self.index_map = []  # [(file_idx, local_idx)]
        self.sample_counts = []

        # Precompute mapping: global index (file_index, sample_index_within_file)
        for file_idx, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as hf:
                num_samples = hf["vil"].shape[0]
                self.sample_counts.append(num_samples)
                for local_idx in range(num_samples):
                    self.index_map.append((file_idx, local_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        with h5py.File(file_path, 'r') as hf:
            vil_np = hf["vil"][sample_idx]  # shape: (20, H, W, 1)
        vil_tensor = torch.from_numpy(vil_np).float()  # convert back to float32 
        return {"vil": vil_tensor}
