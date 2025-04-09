import tensorflow as tf
import os

def check_tfrecords(directory):
    for file in os.listdir(directory):
        if file.endswith('.tfrecords'):
            path = os.path.join(directory, file)
            print(f"Checking {file}...")
            try:
                for _ in tf.data.TFRecordDataset(path, compression_type='GZIP'):
                    pass
            except tf.errors.DataLossError as e:
                print(f"Corrupted: {file} | {e}")
            except Exception as e:
                print(f"Other issue in {file}: {e}")
            else:
                print(f"OK: {file}")

check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data')