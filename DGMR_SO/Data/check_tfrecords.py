import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random

def parse_record(raw_record):
    feature_description = {
        'raw_image_cond': tf.io.FixedLenFeature([], tf.string),
        'raw_image_targ': tf.io.FixedLenFeature([], tf.string),
        'window_cond': tf.io.FixedLenFeature([], tf.float32),
        'height_cond': tf.io.FixedLenFeature([], tf.float32),
        'width_cond': tf.io.FixedLenFeature([], tf.float32),
        'depth_cond': tf.io.FixedLenFeature([], tf.float32),
        'window_targ': tf.io.FixedLenFeature([], tf.float32),
        'height_targ': tf.io.FixedLenFeature([], tf.float32),
        'width_targ': tf.io.FixedLenFeature([], tf.float32),
        'depth_targ': tf.io.FixedLenFeature([], tf.float32),
    }
    return tf.io.parse_single_example(raw_record, feature_description)

def check_black_samples_first_8_frames(tfrecord_dir, pattern="*.tfrecords", value_threshold=0.0, percent_thresh=0.5):
    print(f"Scanning TFRecords in: {tfrecord_dir}")
    tfrecord_files = list(Path(tfrecord_dir).rglob(pattern))

    black_frame_count = 0
    black_sample_count = 0
    total_frames = 0
    total_samples = 0

    for tfrecord_path in tfrecord_files:
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type='GZIP')
            for sample_index, raw_record in enumerate(dataset):
                example = parse_record(raw_record)

                cond = tf.io.parse_tensor(example['raw_image_cond'], out_type=tf.float32).numpy()
                targ = tf.io.parse_tensor(example['raw_image_targ'], out_type=tf.float32).numpy()

                cond_shape = [
                    int(example['window_cond'].numpy()),
                    int(example['height_cond'].numpy()),
                    int(example['width_cond'].numpy()),
                    int(example['depth_cond'].numpy()),
                ]
                targ_shape = [
                    int(example['window_targ'].numpy()),
                    int(example['height_targ'].numpy()),
                    int(example['width_targ'].numpy()),
                    int(example['depth_targ'].numpy()),
                ]

                cond = np.reshape(cond, cond_shape)
                targ = np.reshape(targ, targ_shape)

                # Stack first 8 frames: 4 from cond, 4 from targ
                sample_frames = np.concatenate((cond, targ[:4]), axis=0)

                sample_has_black_frame = False

                for t in range(sample_frames.shape[0]):
                    frame = sample_frames[t, :, :, 0]
                    dark_ratio = np.sum(frame <= value_threshold) / frame.size
                    total_frames += 1
                    if dark_ratio > percent_thresh:
                        print(f"[CHECK] {tfrecord_path.name}, Sample {sample_index}, Frame {t}, dark_ratio: {dark_ratio:.2f}")
                        black_frame_count += 1
                        sample_has_black_frame = True

                if sample_has_black_frame:
                    black_sample_count += 1

                total_samples += 1

        except Exception as e:
            print(f"Error reading {tfrecord_path.name}: {e}")

    print(f"Total samples checked: {total_samples}")
    print(f"Total frames checked (first 8 per sample): {total_frames}")
    print(f"Black (dark) frames found: {black_frame_count}")
    print(f"Samples with ≥1 black frame in first 8: {black_sample_count}")
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

def sanity_check_tfrecords(tfrecord_dir, n_samples=1):
    print(f"Running sanity checks on {tfrecord_dir} ...")
    tfrecord_files = [f for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecords')]
    
    all_records = []
    for file in tfrecord_files:
        path = os.path.join(tfrecord_dir, file)
        try:
            dataset = tf.data.TFRecordDataset([path], compression_type='GZIP')
            count = sum(1 for _ in dataset)
            all_records.extend([(file, i) for i in range(count)])
        except Exception as e:
            print(f"Could not read {file}: {e}")


    sampled_records = random.sample(all_records, min(n_samples, len(all_records)))
    all_pixels = []

    for file, sample_index in sampled_records:
        path = os.path.join(tfrecord_dir, file)
        dataset = tf.data.TFRecordDataset([path], compression_type='GZIP')

        try:
            raw_record = list(dataset.skip(sample_index).take(1))[0]
            example = parse_record(raw_record)

            # Decode and reshape
            cond = tf.io.parse_tensor(example['raw_image_cond'], out_type=tf.float32)
            targ = tf.io.parse_tensor(example['raw_image_targ'], out_type=tf.float32)

            cond_shape = [int(example['window_cond'].numpy()),
                          int(example['height_cond'].numpy()),
                          int(example['width_cond'].numpy()),
                          int(example['depth_cond'].numpy())]

            targ_shape = [int(example['window_targ'].numpy()),
                          int(example['height_targ'].numpy()),
                          int(example['width_targ'].numpy()),
                          int(example['depth_targ'].numpy())]

            cond = tf.reshape(cond, cond_shape).numpy()
            targ = tf.reshape(targ, targ_shape).numpy()

            if np.isnan(cond).any() or np.isnan(targ).any():
                print(f"[{file}] Sample {sample_index}: Contains NaNs")
            if np.isinf(cond).any() or np.isinf(targ).any():
                print(f"[{file}] Sample {sample_index}: Contains Infs")
            if np.all(cond == 0) or np.all(targ == 0):
                print(f"[{file}] Sample {sample_index}: All-zero data")
            if cond.shape != (4, 390, 256, 1) or targ.shape != (16, 390, 256, 1):
                print(f"[{file}] Sample {sample_index}: Unexpected shape {cond.shape}, {targ.shape}")

            all_pixels.append(cond.flatten())
            all_pixels.append(targ.flatten())

        except Exception as e:
            print(f"[{file}] Sample {sample_index}: Error decoding - {e}")

    # Pixel histogram
    if all_pixels:
        pixels = np.concatenate(all_pixels)
        plt.figure(figsize=(8, 4))
        plt.hist(pixels, bins=50, color='skyblue')
        plt.title("Pixel intensity distribution")
        plt.xlabel("Normalized value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__": 
    train_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/train_data'
    val_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/val_data'
    test_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/DGMR-SO/Data/test_data'

    sanity_check_tfrecords(train_data)
    sanity_check_tfrecords(val_data)
    sanity_check_tfrecords(test_data)

    check_tfrecords(train_data)
    check_tfrecords(val_data)
    check_tfrecords(test_data)

    check_black_samples_first_8_frames(tfrecord_dir=train_data, value_threshold=0.0, percent_thresh=0.5)
    check_black_samples_first_8_frames(tfrecord_dir=val_data, value_threshold=0.0, percent_thresh=0.5)
    check_black_samples_first_8_frames(tfrecord_dir=test_data, value_threshold=0.0, percent_thresh=0.5)