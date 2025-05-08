import os
import torch
from netCDFDataset import NetCDFNowcastingDataset

output_dir_train = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data'
output_dir_test = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/test_data'
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

train_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/')
test_dataset = NetCDFNowcastingDataset(root_dir='/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/')
sample_id = 0

for dataset, output_dir in [(train_dataset, output_dir_train), (test_dataset, output_dir_test)]:
    sample_id = 0 
    for idx in range(len(dataset)):
        print(f"Processing sample {idx} of {len(dataset)}")
        try:
            sample = dataset[idx]
            torch.save(sample, os.path.join(output_dir, f"{sample_id:06d}.pt"))
            sample_id += 1
        except IndexError:
            continue
    print(f"Saved {sample_id} valid samples to {output_dir}")

