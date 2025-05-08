import torch
import os

class PreprocessedPTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_dir):
        self.file_paths = sorted([
            os.path.join(pt_dir, fname)
            for fname in os.listdir(pt_dir)
            if fname.endswith(".pt")
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])
