import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from tfrecord_shards_for_nowcasting import Nowcasting_tfrecord

def check_black_samples_like_raw_filter(tfrecord_dir, pattern="*.tfrecords", value_threshold=0.0, percent_thresh=0.5):
    tfrec = Nowcasting_tfrecord()
    dataset = tfrec.get_dataset_large(str(tfrecord_dir), pattern)

    black_frame_count = 0
    black_sample_count = 0
    total_frames = 0
    total_samples = 0

    for sample_index, (cond, targ, mask, date) in enumerate(dataset):
        cond_np = cond.numpy()  # shape: [4, H, W, 1]
        targ_np = targ.numpy()  # shape: [16, H, W, 1]

        sample_has_black_frame = False

        # Check input (condition) frames
        for t in range(cond_np.shape[0]):
            frame = cond_np[t, :, :, 0]
            invalid_mask = (frame <= value_threshold)
            dark_ratio = np.sum(invalid_mask) / frame.size
            total_frames += 1
            if dark_ratio > percent_thresh:
                print(f"[COND] Sample {sample_index}, Frame {t}, dark_ratio: {dark_ratio:.2f}")
                black_frame_count += 1
                sample_has_black_frame = True

        # Check target frames
        for t in range(targ_np.shape[0]):
            frame = targ_np[t, :, :, 0]
            invalid_mask = (frame <= value_threshold)
            dark_ratio = np.sum(invalid_mask) / frame.size
            total_frames += 1
            if dark_ratio > percent_thresh:
                print(f"[TARG] Sample {sample_index}, Frame {t}, dark_ratio: {dark_ratio:.2f}")
                black_frame_count += 1
                sample_has_black_frame = True

        if sample_has_black_frame:
            black_sample_count += 1

        total_samples += 1

    print(f"Total samples checked: {total_samples}")
    print(f"Total frames checked: {total_frames}")
    print(f"Black (dark) frames found: {black_frame_count}")
    print(f"Samples with ≥1 black frame: {black_sample_count}")
    print(f"Percentage of affected samples: {(black_sample_count / total_samples) * 100:.2f}%")


def check_tfrecords(directory):
    corrupted_files = []
    corrupted_files_paths = []
    for file in os.listdir(directory):
        if file.endswith('.tfrecords'):
            path = os.path.join(directory, file)
            print(f"Checking {file}...")
            try:
                for _ in tf.data.TFRecordDataset(path, compression_type='GZIP'):
                    pass
            except tf.errors.DataLossError as e:
                print(f"Corrupted: {file} | {e}")
                corrupted_files.append(file)
                corrupted_files_paths.append(path)
            except Exception as e:
                print(f"Other issue in {file}: {e}")
                corrupted_files.append(file)
                corrupted_files_paths.append(path)
            else:
                print(f"OK: {file}")
    
    for cor in corrupted_files:
        print(f"Corrupted file: {cor}")
        try:
            os.remove(corrupted_files_paths[corrupted_files.index(cor)])
            print(f"Removed corrupted file: {cor}")
        except Exception as e:
            print(f"Failed to remove {cor}: {e}")

#check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data')
#check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data')
#check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data')

check_black_samples_like_raw_filter(
    tfrecord_dir=Path('/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data'), 
    pattern="*.tfrecords",
    value_threshold=0.0,
    percent_thresh=0.5
)

check_black_samples_like_raw_filter(
    tfrecord_dir=Path('/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data'), 
    pattern="*.tfrecords",
    value_threshold=0.0,
    percent_thresh=0.5
)

check_black_samples_like_raw_filter(
    tfrecord_dir=Path('/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data'), 
    pattern="*.tfrecords",
    value_threshold=0.0,
    percent_thresh=0.5
)
