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
    Get all the files from the path with muti-folders
    """
    all_file_list = os.listdir(path)
    all_file_list.sort()
    # Go through all the folders
    for file in all_file_list:
        file_path = os.path.join(path, file)
     
        if os.path.isdir(file_path):
            get_all_files(file_path)
        elif os.path.isfile(file_path) and file.startswith(('.')) == False:
            all_file_full_path_list.append(file_path)
            all_file_name_list.append(file)
            datestr = file.split('_')[2].split('.')[0]
            startTime = datetime.datetime.strptime(datestr, "%Y%m%d%H%M")
            label_list.append(startTime)
    return all_file_full_path_list, all_file_name_list,label_list


def write_tfrecord(INPUT_PATH, batches, windows, height, width, folder, output_path):
    all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)
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
                # print('Skip files start from: ' + str(label_list[id + i]))
                continue
            nc_file_start = Dataset(all_file_full_path_list[id + i])
            nc_file_end = Dataset(all_file_full_path_list[id + i + windows])
            start_time = label_list[id + i]
            end_time = label_list[id + i + 3]
            # Remove night time data
            if start_time.hour < 5 or start_time.hour > 17:
                continue
            if np.any(nc_file_start['sds'][:] <= 0):
                # print('remove start ' + str(label_list[id + i]))
                continue
            if np.any(nc_file_end['sds'][:] <= 0):
                # print('remove end ' + str(label_list[id + i + windows]))
                continue

            # Nomralize SDS using SDS_CS
            for j in range(windows):
                nc_file = Dataset(all_file_full_path_list[id + i + j])
                ele_sds = nc_file.variables['sds'][:]
                ele_cs = nc_file.variables['sds_cs'][:]
                ele_cs[ele_cs < 0] = 0
                ele = ele_sds / ele_cs
                ele[np.isnan(ele)] = 0

                if np.any(ele <= 0):
                    print('Data is partly 0: ' + str(label_list[id + i + j]) + 'window is: ' + str(j))
                if np.all(ele <= 0):
                    print('Attention!!!: ' + str(label_list[id + i + j]) + 'window is: ' + str(j))
                # Note that here it is (width, heigh) while the tensor is in (rows = height, cols = width)
                dataset[j] = np.array(ele.T)
                
            # Convert data to tensot format
            # Windows, height, width, depth
            dataset_3D = np.expand_dims(dataset, axis=-1) 
            dataset = np.zeros(shape=(windows, height, width))
            dataset_x, dataset_y = split_data_xy_1(dataset_3D)

            # Current time:forecast start time
            date_y = label_list[id + i + 4: id + i + windows]
            total_data = total_data + 1
            print("total valid sample by far is: " + str(total_data))

            # Rolling 10 steps in each .tfrecord file
            tf_record_var.write_images_to_tfr_long(dataset_x.astype('float32'), dataset_y.astype('float32'), str(date_y))
            
        tf_record_var = Nowcasting_tfrecord(output_path, str(id) + folder, batches, 0)
        id = id + batches

if __name__ == "__main__":
    TRAIN_INPUT_PATH = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_train_data/'
    VAL_INPUT_PATH = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_val_data/'
    TEST_INPUT_PATH = '/net/pc200258/nobackup_1/users/meirink/Jeroen/raw_test_data/'
    OUTPUT_PATH_train = Path('./Data/train_data')
    OUTPUT_PATH_val = Path('./Data/val_data')
    OUTPUT_PATH_test = Path('./Data/test_data')

    batches = 200
    windows = 20
    height = 256 # Matches the corresponding area
    width = 390 # Matches the corresponding area

    data_array = ['train', 'val', 'test']
    for data in data_array:
        if data == 'train':
            folder = "_train"
            write_tfrecord(TRAIN_INPUT_PATH, batches, windows, height, width, folder, OUTPUT_PATH_train)
        elif data == 'val':
            folder = "_val"
            write_tfrecord(VAL_INPUT_PATH, batches, windows, height, width, folder, OUTPUT_PATH_val)
        elif data == 'test':
            folder = "_test"
            write_tfrecord(TEST_INPUT_PATH, batches, windows, height, width, folder, OUTPUT_PATH_test)
