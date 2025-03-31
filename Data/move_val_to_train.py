import os
import shutil
import datetime

val_data_path = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_val_data/'
train_data_root = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data/'

# Move files back to appropriate year folder
files = [f for f in os.listdir(val_data_path) if f.endswith('.nc')]

for file in files:
    try:
        datestr = file.split('_')[8]  
        dt = datetime.datetime.strptime(datestr, "%Y%m%dT%H%M%S")
        year = str(dt.year)

        # Build destination path
        dest_folder = os.path.join(train_data_root, year)
        src = os.path.join(val_data_path, file)
        dest = os.path.join(dest_folder, file)

        # Move the file
        shutil.move(src, dest)
        print(f"Moved {file} → {year}/")
    except Exception as e:
        print(f"Skipping {file}: {e}")

print("Finished moving validation files back to training folders.")
