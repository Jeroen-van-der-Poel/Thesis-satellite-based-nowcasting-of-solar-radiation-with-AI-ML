from netCDF4 import Dataset
import os
import numpy as np
import datetime
from tfrecord_shards_for_nowcasting import Nowcasting_tfrecord
from pathlib import Path

all_file_full_path_list = []
all_file_name_list = []
label_list = []

# Split the dataset into 4 past frames (X) and 16 future frames (Y)
def split_data_xy_1(data):
    x = data[0:4, :, :, :]
    y = data[4:20,:, :, :]
    return x, y

def get_all_files(path):
    """
    Get all the files from the path including subfolders.
    """
    all_file_full_path_list = []
    all_file_name_list = []
    label_list = []

    for root, _, files in os.walk(path):
        files.sort()
        for file in files:
            if not file.startswith('.'):
                file_path = os.path.join(root, file)
                all_file_full_path_list.append(file_path)
                all_file_name_list.append(file)
                try:
                    datestr = file.split('_')[8] 
                    startTime = datetime.datetime.strptime(datestr, "%Y%m%dT%H%M%S")
                    label_list.append(startTime)
                except Exception as e:
                    print(f"Skipping file {file} due to error: {e}")

    return all_file_full_path_list, all_file_name_list, label_list

def write_tfrecord(INPUT_PATH, batches, windows, height, width, folder, output_path):
    all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)

    # Sort the files based on the timestamp
    full_list = list(zip(label_list, all_file_full_path_list, all_file_name_list))
    full_list.sort(key=lambda x: x[0])
    label_list, all_file_full_path_list, all_file_name_list = zip(*full_list)
    label_list = list(label_list)
    all_file_full_path_list = list(all_file_full_path_list)
    all_file_name_list = list(all_file_name_list)

    dataset = np.zeros(shape=(windows, height, width))
    length = len(label_list)
    id = 0
    exact_end_time = datetime.timedelta(minutes=(windows * 15))
    tf_record_var = Nowcasting_tfrecord(output_path, str(id) + folder, batches, 0)
    total_data = 0
    while id < length:
        print(id)
        print(label_list[id], ', Progress: ' + str(id) + '/' + str(length))
        # In case some data is missing, then skip this window and expand the window
        # Make sure there are 20 data in each sample
        if id + windows >= length:
            break
        # Loop for samples in each tfrecord
        for i in range(batches):
            if id + i + windows >= length:
                break
            interval = label_list[id + i + windows] - label_list[id + i]
            if interval != exact_end_time:
                id = id + 1
                print('Skip files start from: ' + str(label_list[id + i]))
                continue

            start_time = label_list[id + i]
            end_time = label_list[id + i + 3]

            with Dataset(all_file_full_path_list[id + i]) as nc_file_start, Dataset(all_file_full_path_list[id + i + windows]) as nc_file_end:
                if np.any(nc_file_start.variables['sds'][0,:,:] <= 0):
                    print('remove start ' + str(label_list[id + i]))
                    continue
                if np.any(nc_file_end.variables['sds'][0,:,:] <= 0):
                    print('remove end ' + str(label_list[id + i + windows]))
                    continue

            too_dark = False
            # Normalize SDS using SDS_CS
            for j in range(windows):
                with Dataset(all_file_full_path_list[id + i + j]) as nc_file:

                    # Check for night coverage (> 50% NaN or -1) in the first 8 frames
                    if j < 8:
                        ele_sds_raw = nc_file.variables['sds'][0, :, :]
                        ele_sds_new = ele_sds_raw.filled(-1) 
                        total_pixels = ele_sds_new.size
                        invalid_mask = np.logical_or(np.isnan(ele_sds_new), ele_sds_new <= 0)
                        dark_ratio = np.sum(invalid_mask) / total_pixels
                        if dark_ratio > 0.5:
                            print(f"Skipping sample due to dark image at window frame {j}: {label_list[id + i + j]} (dark_ratio={dark_ratio:.2f})")
                            too_dark = True
                            break
                    
                    ele_sds = nc_file.variables['sds'][0,:,:]
                    ele_cs = nc_file.variables['sds_cs'][0,:,:]
                    ele_cs[ele_cs < 0] = 0
                    ele = ele_sds / ele_cs
                    ele[np.isnan(ele)] = 0
                    ele[ele < 0] = 0

                    if np.any(ele <= 0):
                        print('Data is partly 0: ' + str(label_list[id + i + j]) + 'window is: ' + str(j))
                    if np.all(ele <= 0):
                        print('Attention!!!: ' + str(label_list[id + i + j]) + 'window is: ' + str(j))
                    # Note that here it is (width, heigh) while the tensor is in (rows = height, cols = width)
                    dataset[j] = np.array(ele.T)
            
            if too_dark:
                continue
                
            # Convert data to tensot format
            # Windows, height, width, depth
            dataset_3D = np.expand_dims(dataset, axis=-1) 
            dataset = np.zeros(shape=(windows, height, width))
            dataset_x, dataset_y = split_data_xy_1(dataset_3D)

            # Current time:forecast start time
            date_y = label_list[id + i + 4: id + i + windows]
            total_data = total_data + 1
            print("Total valid sample so far is: " + str(total_data))

            # Rolling 10 steps in each .tfrecord file
            tf_record_var.write_images_to_tfr_long(dataset_x.astype('float32'), dataset_y.astype('float32'), str(date_y))
            
        tf_record_var = Nowcasting_tfrecord(output_path, str(id) + folder, batches, 0)
        id = id + batches

if __name__ == "__main__":
    TRAIN_INPUT_PATH = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data/'
    TEST_INPUT_PATH = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_test_data/'
    OUTPUT_PATH_train = Path('/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/train_data')
    OUTPUT_PATH_test = Path('/net/pc200258/nobackup_1/users/meirink/Jeroen/Thesis-satellite-based-nowcasting-of-solar-radiation-with-AI-ML/Data/test_data')

    batches = 200
    windows = 20
    height = 390 # Matches the corresponding area
    width = 256 # Matches the corresponding area

    data_array = ['train', 'test']
    for data in data_array:
        if data == 'train':
            folder = "_train"
            write_tfrecord(TRAIN_INPUT_PATH, batches, windows, height, width, folder, OUTPUT_PATH_train)
        elif data == 'test':
            folder = "_test"
            write_tfrecord(TEST_INPUT_PATH, batches, windows, height, width, folder, OUTPUT_PATH_test)
