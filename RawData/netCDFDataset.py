import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as NetCDF
import numpy as np
import os
import datetime
import gc
from multiprocessing import Manager

class NetCDFNowcastingDataset(Dataset):
    def __init__(self, root_dir, window=20, x_frames=4, y_frames=16, height=390, width=256):
        manager = Manager()
        self.root_dir = root_dir
        self.window = window
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.height = height
        self.width = width
        self.file_paths, self.timestamps = manager.list(self._load_all_files_sorted())
        self.valid_indices = manager.list(self._filter_valid_indices())

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
        return np.array(paths), np.array(times)

    def _filter_valid_indices(self):
        valid = []
        expected_interval = datetime.timedelta(minutes=(self.window - 1) * 15)
        for idx in range(len(self.file_paths) - self.window):
            try:
                start_time = self.timestamps[idx]
                end_time = self.timestamps[idx + self.window - 1]
                if end_time - start_time != expected_interval:
                    continue

                too_dark = False
                for i in range(self.window):
                    with NetCDF(self.file_paths[idx + i]) as nc:
                        sds = nc.variables['sds'][0, :, :]
                        if i < 8:
                            sds_new = sds.filled(-1) if hasattr(sds, 'filled') else sds
                            total_pixels = sds_new.size
                            invalid_mask = np.logical_or(np.isnan(sds_new), sds_new <= 0)
                            dark_ratio = np.sum(invalid_mask) / total_pixels
                            if dark_ratio > 0.5:
                                print(f"Too dark at {self.timestamps[idx + i]} (dark_ratio={dark_ratio:.2f})")
                                too_dark = True
                                break

                if not too_dark:
                    valid.append(idx)
            except Exception as e:
                print(f"Skipping index {idx} due to error: {e}")
                continue

        print(f"Filtered {len(valid)} valid samples.")
        return np.array(valid, dtype=np.int32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):   #xarray
        idx = self.valid_indices[i]
        x = torch.zeros((self.x_frames, self.height, self.width), dtype=torch.float32)
        y = torch.zeros((self.y_frames, self.height, self.width), dtype=torch.float32)
        x_cs = torch.zeros((self.x_frames, self.height, self.width), dtype=torch.float32)
        y_cs = torch.zeros((self.y_frames, self.height, self.width), dtype=torch.float32)

        for j in range(self.window):
            with NetCDF(self.file_paths[idx + j]) as nc:
                sds = nc.variables['sds'][0, :, :]
                sds_cs = nc.variables['sds_cs'][0, :, :]

                sds_cs[sds_cs < 0] = 0
                norm = np.clip(sds / sds_cs, 0, 1).T
                cs = sds_cs.T

                norm_tensor = torch.from_numpy(norm).float()
                cs_tensor = torch.from_numpy(cs).float()

                if j < self.x_frames:
                    x[j] = norm_tensor
                    x_cs[j] = cs_tensor
                else:
                    y[j - self.x_frames] = norm_tensor
                    y_cs[j - self.x_frames] = cs_tensor

        norm_tensor  = torch.cat([x, y], dim=0).unsqueeze(-1)
        cs_tensor = torch.cat([x_cs, y_cs], dim=0).unsqueeze(-1)

        # Help GC
        del x, y, x_cs, y_cs, norm, sds, sds_cs, cs
        gc.collect()  
        torch.cuda.empty_cache()

        return norm_tensor, cs_tensor

    def count_valid_samples(self):
        print(f"Total valid samples: {len(self.valid_indices)}")
        return len(self.valid_indices)
