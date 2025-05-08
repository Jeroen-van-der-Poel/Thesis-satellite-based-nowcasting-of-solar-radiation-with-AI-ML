import os
import random
import shutil

# Train data path
train_tfrecord_path = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/train_data'

# Validation output path
val_tfrecord_path = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/val_data'
os.makedirs(val_tfrecord_path, exist_ok=True)

# Get all TFRecord files
tfrecord_files = [f for f in os.listdir(train_tfrecord_path) if f.endswith('.tfrecords')]

# Shuffle and select 20%
random.seed(42)
random.shuffle(tfrecord_files)
val_count = int(0.20 * len(tfrecord_files))
val_files = tfrecord_files[:val_count]

# Move and rename files
for f in val_files:
    src = os.path.join(train_tfrecord_path, f)
    new_filename = f.replace('_train', '_val')
    dst = os.path.join(val_tfrecord_path, new_filename)
    try:
        shutil.move(src, dst)
    except Exception as e:
        print(f"Error moving {f}: {e}")

print(f"Moved {len(val_files)} TFRecord files from train to val (with renaming).")
