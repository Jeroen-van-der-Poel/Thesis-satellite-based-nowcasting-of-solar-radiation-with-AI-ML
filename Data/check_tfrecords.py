import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from tfrecord_shards_for_nowcasting import Nowcasting_tfrecord

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

def check_black_samples(tfrecord_dir, pattern="*.tfrecords", value_threshold=0.0, percent_thresh=0.5):
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

                sample_has_black_frame = False

                # Check cond frames
                for t in range(cond.shape[0]):
                    frame = cond[t, :, :, 0]
                    dark_ratio = np.sum(frame <= value_threshold) / frame.size
                    total_frames += 1
                    if dark_ratio > percent_thresh:
                        print(f"[COND] {tfrecord_path.name}, Sample {sample_index}, Frame {t}, dark_ratio: {dark_ratio:.2f}")
                        black_frame_count += 1
                        sample_has_black_frame = True

                # Check targ frames
                for t in range(targ.shape[0]):
                    frame = targ[t, :, :, 0]
                    dark_ratio = np.sum(frame <= value_threshold) / frame.size
                    total_frames += 1
                    if dark_ratio > percent_thresh:
                        print(f"[TARG] {tfrecord_path.name}, Sample {sample_index}, Frame {t}, dark_ratio: {dark_ratio:.2f}")
                        black_frame_count += 1
                        sample_has_black_frame = True

                if sample_has_black_frame:
                    black_sample_count += 1

                total_samples += 1

        except Exception as e:
            print(f"Error reading {tfrecord_path.name}: {e}")

    print(f"Total samples checked: {total_samples}")
    print(f"Total frames checked: {total_frames}")
    print(f"Black (dark) frames found: {black_frame_count}")
    print(f"Samples with ≥1 black frame: {black_sample_count}")
    print(f"Percentage of affected samples: {(black_sample_count / total_samples) * 100:.2f}%")

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

check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data')
check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data')
check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data')

# check_black_samples(
#     tfrecord_dir=Path('/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data'), 
#     pattern="*.tfrecords",
#     value_threshold=0.0,
#     percent_thresh=0.5
# )

# check_black_samples(
#     tfrecord_dir=Path('/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data'), 
#     pattern="*.tfrecords",
#     value_threshold=0.0,
#     percent_thresh=0.5
# )

# check_black_samples(
#     tfrecord_dir=Path('/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data'), 
#     pattern="*.tfrecords",
#     value_threshold=0.0,
#     percent_thresh=0.5
# )

# check_black_samples_first_8_frames(
#     tfrecord_dir='/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data',
#     value_threshold=0.0,
#     percent_thresh=0.5
# )

# check_black_samples_first_8_frames(
#     tfrecord_dir='/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data',
#     value_threshold=0.0,
#     percent_thresh=0.5
# )
# check_black_samples_first_8_frames(
#     tfrecord_dir='/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data',
#     value_threshold=0.0,
#     percent_thresh=0.5
# )