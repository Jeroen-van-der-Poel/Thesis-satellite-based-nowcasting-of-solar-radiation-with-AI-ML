import os
import glob
import random
import shutil
from pathlib import Path

# Input directories (excluding 2020)
input_root = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data/'
years = ['2021', '2022', '2023', '2024']

# Output directory for validation files
val_output_path = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_val_data/'
if not os.path.exists(val_output_path):
    os.makedirs(val_output_path)

# Gather all file paths from the specified years
all_files = []
for year in years:
    year_path = os.path.join(input_root, year)
    files = glob.glob(os.path.join(year_path, '*.nc'))
    all_files.extend(files)

print(f"Total training files (excluding 2020): {len(all_files)}")

# Shuffle and select 20%
random.seed(42) 
random.shuffle(all_files)
val_count = int(0.20 * len(all_files))
val_files = all_files[:val_count]

print(f"Selected {val_count} files for validation.")

for f in val_files:
    shutil.move(f, val_output_path)

print("Validation split done.")
