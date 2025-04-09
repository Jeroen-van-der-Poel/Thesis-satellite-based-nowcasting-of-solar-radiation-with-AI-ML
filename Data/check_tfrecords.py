import tensorflow as tf
import os

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
        # try:
        #     os.remove(corrupted_files_paths[corrupted_files.index(cor)])
        #     print(f"Removed corrupted file: {cor}")
        # except Exception as e:
        #     print(f"Failed to remove {cor}: {e}")

#check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data')
#check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/val_data')
check_tfrecords('/data1/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data')