import os
import torch
from netCDFDataset import NetCDFNowcastingDataset

output_dir_train = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data'
output_dir_test = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data'
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

train_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/')
test_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/')

samples_per_pt = 500

def save_batched_pt(dataset, output_dir, batch_size):
    sample_id = 0
    batch = []
    total_valid_samples = 0
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            batch.append(sample)
            total_valid_samples += 1
            if len(batch) == batch_size:
                torch.save(batch, os.path.join(output_dir, f"{sample_id:06d}.pt"), _use_new_zipfile_serialization=True)
                sample_id += 1
                batch = []
        except IndexError:
            continue
    # Save any remaining samples
    if batch:
        torch.save(batch, os.path.join(output_dir, f"{sample_id:06d}.pt"))

    print(f"Saved {sample_id + (1 if batch else 0)} batched files to {output_dir}")
    print(f"Total valid samples saved: {total_valid_samples}")


save_batched_pt(train_dataset, output_dir_train, samples_per_pt)
save_batched_pt(test_dataset, output_dir_test, samples_per_pt)