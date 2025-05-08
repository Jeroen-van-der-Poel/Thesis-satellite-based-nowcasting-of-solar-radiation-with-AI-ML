import torch
import os

class PreprocessedPTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_dir):
        self.samples = []
        for fname in sorted(os.listdir(pt_dir)):
            if fname.endswith(".pt"):
                path = os.path.join(pt_dir, fname)
                batch = torch.load(path)
                self.samples.extend(batch)  # Unpack the list of samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
