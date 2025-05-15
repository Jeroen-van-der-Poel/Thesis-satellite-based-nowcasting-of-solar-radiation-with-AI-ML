import os
import tensorflow as tf
from utils.tfrecord_shards_for_nowcasting import Nowcasting_tfrecord
import matplotlib.pyplot as plt
import random
import numpy as np 

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
