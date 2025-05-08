import os
import shutil
import random

train_data_dir = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/train_data"
val_data_dir = "/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/EarthFormer/Data/val_data"

os.makedirs(val_data_dir, exist_ok=True)

all_files = [f for f in os.listdir(train_data_dir) if f.endswith(".pt")]
all_files.sort() 
random.seed(42)
random.shuffle(all_files)

# Split 80/20
split_idx = int(0.8 * len(all_files))
train_files, val_files = all_files[:split_idx], all_files[split_idx:]

# Move validation files
for fname in val_files:
    src = os.path.join(train_data_dir, fname)
    dst = os.path.join(val_data_dir, fname)
    shutil.move(src, dst)

print(f"Split complete: {len(train_files)} training files remain, {len(val_files)} moved to validation.")
