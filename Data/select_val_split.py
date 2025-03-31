import os
import glob
import random
import shutil
import datetime

# Input directories (excluding 2020)
input_root = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data/'
years = ['2021', '2022', '2023', '2024']

# Output directory for validation files
val_output_path = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_val_data/'
if not os.path.exists(val_output_path):
    os.makedirs(val_output_path)

# Load all .nc files with timestamps
file_timestamp_pairs = []

for year in years:
    year_path = os.path.join(input_root, year)
    files = glob.glob(os.path.join(year_path, '*.nc'))
    for f in files:
        try:
            filename = os.path.basename(f)
            datestr = filename.split('_')[8]  # adjust if different format
            dt = datetime.datetime.strptime(datestr, "%Y%m%dT%H%M%S")
            file_timestamp_pairs.append((f, dt))
        except Exception as e:
            print(f"Skipping {f}, reason: {e}")

# Sort by timestamp
file_timestamp_pairs.sort(key=lambda x: x[1])

# Create a dict for fast access
file_dict = {dt: f for f, dt in file_timestamp_pairs}
timestamps = [dt for _, dt in file_timestamp_pairs]

# Find valid 5-hour windows (20 steps of 15 min)
valid_windows = []
step = datetime.timedelta(minutes=15)
window_len = 20

for i in range(len(timestamps) - window_len + 1):
    valid = True
    for j in range(1, window_len):
        if timestamps[i + j] - timestamps[i + j - 1] != step:
            valid = False
            break
    if valid:
        window = [file_dict[timestamps[i + k]] for k in range(window_len)]
        valid_windows.append(window)

print(f"Total valid 5-hour windows found: {len(valid_windows)}")

# Shuffle and take 20%
random.seed(42)
random.shuffle(valid_windows)
val_window_count = int(0.20 * len(valid_windows))
val_windows = valid_windows[:val_window_count]

# Move files to validation folder
for window in val_windows:
    for f in window:
        try:
            shutil.move(f, os.path.join(val_output_path, os.path.basename(f)))
        except Exception as e:
            print(f"Error moving {f}: {e}")

print(f"Moved {val_window_count} 5-hour sequences to validation folder.")
