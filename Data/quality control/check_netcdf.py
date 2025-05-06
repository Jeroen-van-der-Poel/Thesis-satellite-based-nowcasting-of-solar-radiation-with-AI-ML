from torch.utils.data import DataLoader
import torch
import numpy as np
from Data.netCDFDataset import NetCDFNowcastingDataset  
from tqdm import tqdm

DATA_PATH = "/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data/"
BATCH_SIZE = 1  
MAX_CHECK = 500  

dataset = NetCDFNowcastingDataset(root_dir=DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def check_tensor_range(tensor, name=""):
    if not (torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0)):
        print(f"[Range ERROR] {name} out of [0, 1] range!")
        return False
    return True

def check_darkness(x_tensor):
    x_np = x_tensor.numpy()
    for i in range(min(8, x_np.shape[0])):  # only first 8 frames
        dark_pixels = np.logical_or(np.isnan(x_np[i]), x_np[i] <= 0)
        dark_ratio = np.sum(dark_pixels) / dark_pixels.size
        if dark_ratio > 0.5:
            print(f"[Darkness ERROR] Frame {i} is too dark ({dark_ratio:.2f})")
            return False
    return True

# Run checks
valid_sample_count = 0
for idx, (x, y) in enumerate(tqdm(loader, desc="Validating dataset")):
    x, y = x.squeeze(), y.squeeze()

    if check_tensor_range(x, "Input") and check_tensor_range(y, "Target") and check_darkness(x):
        valid_sample_count += 1
    else:
        print(f"Issue at sample index: {idx}")
        continue 

    if MAX_CHECK is not None and idx >= MAX_CHECK:
        break

print(f"Total valid rolling window samples: {valid_sample_count}")
