import h5py
import numpy as np
from data_preperation_to_tfr import get_all_keys_from_h5, get_all_files

INPUT_PATH = './202112/'

all_file_full_path_list = []
all_file_name_list = []
label_list = []

all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)

length = len(label_list)
max = 0
for i in range(length):
    h5_file = h5py.File(all_file_full_path_list[i])
    keys = get_all_keys_from_h5(h5_file)
    sds = h5_file[keys[3]][:]
    for j in range(8):
        if j == 3:
            ele = h5_file[keys[j]][:]
            value = np.max(ele)
            if value > max:
                max = value
            else:
                continue
print("max is: " + str(max))