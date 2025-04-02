import os
import glob
import tensorflow as tf
import datetime
from tfrecord_shards_for_nowcasting import Nowcasting_tfrecord
import matplotlib.pyplot as plt
import random

def get_all_nc_files(path):
    all_file_name_list = []
    for root, _, files in os.walk(path):
        files.sort()
        for file in files:
            if not file.startswith('.'):
                all_file_name_list.append(file)
    return all_file_name_list

def count_tfrecord_files(tfrecord_dir):
    total_samples = 0
    tfrecord_files = [f for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecords')]

    for tfrecord_file in tfrecord_files:
        path = os.path.join(tfrecord_dir, tfrecord_file)
        try:
            for _ in tf.data.TFRecordDataset([path], compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE):
                total_samples += 1
        except tf.errors.DataLossError as e:
            print(f"Skipped corrupted file: {tfrecord_file} | Error: {e}")
    return total_samples

def get_directory_size_in_gb(path):
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    return round(total_bytes / (1024**3), 2)  

def visualize_random_sample(tfrecord_path, title):
    parser = Nowcasting_tfrecord()
    
    dataset = parser.get_dataset_large(tfrecord_path, pattern="*.tfrecords", has_prob=True)

    for cond, targ, mask, prob, date in dataset.shuffle(10).take(1): 
        cond = cond.numpy()
        targ = targ.numpy()

        plt.figure(figsize=(16, 4))
        
        for i in range(4):  # Visualize 4 past frames
            plt.subplot(2, 4, i + 1)
            plt.imshow(cond[i, :, :, 0], cmap='gray')
            plt.title(f'Cond t-{4 - i}')
            plt.axis('off')

        for i in range(4):  # Visualize 4 future frames
            plt.subplot(2, 4, i + 5)
            plt.imshow(targ[i, :, :, 0], cmap='gray')
            plt.title(f'Target t+{i + 1}')
            plt.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        break 

if __name__ == "__main__":
    raw_train_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data'
    raw_val_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_val_data'
    raw_test_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_test_data'
    
    train_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data'
    val_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data'
    test_data = '/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data'

    # Takes a very long time to calculate
    # total_train_samples = count_tfrecord_files(train_data)
    # total_val_samples = count_tfrecord_files(val_data)
    # total_test_samples = count_tfrecord_files(test_data)

    total_raw_data = len(get_all_nc_files(raw_train_data)) + len(get_all_nc_files(raw_val_data)) + len(get_all_nc_files(raw_test_data))
    print(f"Total raw data: {total_raw_data}")
    print(f"Total raw train data: {len(get_all_nc_files(raw_train_data))}")
    print(f"Total raw val data: {len(get_all_nc_files(raw_val_data))}")
    print(f"Total raw test data: {len(get_all_nc_files(raw_test_data))}")

    # print(f"Total new train data: {total_train_samples}")
    # print(f"Total new val data: {total_val_samples}")
    # print(f"Total new test data: {total_test_samples}")

    print(get_directory_size_in_gb(train_data))
    print(get_directory_size_in_gb(val_data))
    print(get_directory_size_in_gb(test_data))

    print("Visualizing random train sample...")
    visualize_random_sample(train_data, title="Train Sample")
    print("Visualizing random val sample...")
    visualize_random_sample(val_data, title="Validation Sample")
    print("Visualizing random test sample...")
    visualize_random_sample(test_data, title="Test Sample")