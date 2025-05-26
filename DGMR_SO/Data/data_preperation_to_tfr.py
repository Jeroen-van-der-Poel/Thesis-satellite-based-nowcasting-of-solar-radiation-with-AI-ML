import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import random
import shutil
import numpy as np
from utils.tfrecord_shards_for_nowcasting import Nowcasting_tfrecord
from pathlib import Path
from RawData.netCDFDataset import NetCDFNowcastingDataset

def write_tfrecord(input_path, batches, folder, output_path):
    dataset = NetCDFNowcastingDataset(input_path)

    tf_record_writer = None
    total_samples = len(dataset)
    print(f"Total valid samples to write: {total_samples}")

    for i in range(total_samples):
        if i % batches == 0:
            if tf_record_writer:
                del tf_record_writer   # Close the previous writer
            tf_record_writer = Nowcasting_tfrecord(output_path, f"{i}{folder}", batches, 0)

        sample = dataset[i]["vil"].numpy()  # Shape: (20, H, W, 1)
        x = sample[:4]   # Shape: (4, H, W, 1)
        y = sample[4:]   # Shape: (16, H, W, 1)

        timestamps = [str(dataset.timestamps[dataset.valid_indices[i] + j]) for j in range(4, 20)]

        tf_record_writer.write_images_to_tfr_long(x.astype("float32"), y.astype("float32"), timestamps)

        if (i + 1) % 10 == 0 or i == total_samples - 1:
            print(f"Written {i + 1}/{total_samples} samples")

def split_train_to_val(train_path, val_path, val_ratio=0.2, rename=True):
    os.makedirs(val_path, exist_ok=True)
    tfrecord_files = [f for f in os.listdir(train_path) if f.endswith('.tfrecords')]

    random.seed(42)
    random.shuffle(tfrecord_files)
    val_count = int(val_ratio * len(tfrecord_files))
    val_files = tfrecord_files[:val_count]

    for f in val_files:
        src = os.path.join(train_path, f)
        dst = os.path.join(val_path, f.replace('_train', '_val') if rename else f)
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"Error moving {f}: {e}")

    print(f"Moved {len(val_files)} TFRecord files to validation folder.")

if __name__ == "__main__":
    TRAIN_INPUT_PATH = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_train_data/'
    TEST_INPUT_PATH = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/RawData/raw_test_data/'
    OUTPUT_PATH_train = Path('/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/train_data')
    OUTPUT_PATH_val = Path('/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/val_data')
    OUTPUT_PATH_test = Path('/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/test_data')

    batches = 200

    for data, path, output in [("train", TRAIN_INPUT_PATH, OUTPUT_PATH_train), ("test", TEST_INPUT_PATH, OUTPUT_PATH_test)]:
        folder = f"_{data}"
        write_tfrecord(path, batches, folder, output)
    
    split_train_to_val(OUTPUT_PATH_train, OUTPUT_PATH_val, val_ratio=0.2, rename=True)

