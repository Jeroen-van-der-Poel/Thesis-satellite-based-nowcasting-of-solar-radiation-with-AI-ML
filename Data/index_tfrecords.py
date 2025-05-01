from tfrecord.tools import tfrecord2idx
import os
from glob import glob

# This script indexes TFRecord files in a specified directory. Only use this for EarthFormer not needed for DGMR_SO

def index_tfrecords(tfrecord_dir):
    tfrecord_files = glob(os.path.join(tfrecord_dir, '*.tfrecords'))
    for tf_file in tfrecord_files:
        index_file = tf_file + '.index'
        if not os.path.exists(index_file):
            print(f"Indexing {tf_file}...")
            tfrecord2idx.create_index(tf_file, index_file, compression_type='GZIP')
        else:
            print(f"Index already exists for {tf_file}")

def remove_index_files(tfrecord_dir):
    index_files = glob(os.path.join(tfrecord_dir, '*.tfrecords.index'))
    for idx_file in index_files:
        print(f"Removing {idx_file}")
        os.remove(idx_file)

if __name__ == "__main__":
    #index_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data')
    remove_index_files('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data')
    index_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data')
    #index_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data')