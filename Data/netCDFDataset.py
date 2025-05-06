import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as NetCDF
import numpy as np
import os
import datetime
from torch.utils.data import DataLoader

class NetCDFNowcastingDataset(Dataset):
    def __init__(self, root_dir, window=20, x_frames=4, y_frames=16, height=390, width=256):
        self.root_dir = root_dir
        self.window = window
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.height = height
        self.width = width
        self.file_paths, self.timestamps = self._load_all_files_sorted()

    def _load_all_files_sorted(self):
        paths, times = [], []
        for root, _, files in os.walk(self.root_dir):
            for fname in sorted(files):
                if fname.startswith('.'):
                    continue
                full_path = os.path.join(root, fname)
                try:
                    t = datetime.datetime.strptime(fname.split('_')[8], "%Y%m%dT%H%M%S")
                    paths.append(full_path)
                    times.append(t)
                except Exception:
                    continue
        sorted_pairs = sorted(zip(times, paths))
        times, paths = zip(*sorted_pairs)
        return list(paths), list(times)

    def __len__(self):
        return len(self.file_paths) - self.window

    def __getitem__(self, idx):
        while idx < len(self) - self.window:
            x = np.zeros((self.x_frames, self.height, self.width), dtype=np.float32)
            y = np.zeros((self.y_frames, self.height, self.width), dtype=np.float32)
            too_dark = False

            for i in range(self.window):
                with NetCDF(self.file_paths[idx + i]) as nc:
                    sds = nc.variables['sds'][0, :, :]
                    sds_cs = nc.variables['sds_cs'][0, :, :]
                    sds_cs[sds_cs < 0] = 0
                    norm = sds / np.maximum(sds_cs, 1e-6)
                    #norm[np.isnan(norm)] = 0
                    #norm[norm < 0] = 0
                    norm = np.clip(norm, 0, 1)  # Ensure values are between 0 and 1
                    norm = norm.T  # (H, W)

                    if i < 8:
                        sds_new = sds.filled(-1) 
                        total_pixels = sds_new.size
                        invalid_mask = np.logical_or(np.isnan(sds_new), sds_new <= 0)
                        dark_ratio = np.sum(invalid_mask) / total_pixels
                        if dark_ratio > 0.5:
                            too_dark = True
                            break

                    if i < self.x_frames:
                        x[i] = norm
                    else:
                        y[i - self.x_frames] = norm

            if too_dark:
                idx += 1
                continue  # Try next index
            else:
                return torch.from_numpy(x), torch.from_numpy(y)

        raise IndexError("No valid sample found from this index onward.")



# Update this to point to your NetCDF root directory
root_dir = "/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data/"

# Instantiate dataset
dataset = NetCDFNowcastingDataset(root_dir=root_dir)

# Print dataset length
print(f"Dataset contains {len(dataset)} samples.")

# Load one sample
x, y = dataset[0]
print("Input shape:", x.shape)  # Expected: (4, 390, 256)
print("Target shape:", y.shape)  # Expected: (16, 390, 256)

# Check value ranges
print("Input min/max:", x.min().item(), x.max().item())
print("Target min/max:", y.min().item(), y.max().item())
