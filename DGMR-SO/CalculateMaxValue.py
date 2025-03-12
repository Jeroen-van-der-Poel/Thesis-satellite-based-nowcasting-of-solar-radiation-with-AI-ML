import math
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import numbers

INPUT_PATH = './202112/'

def get_all_keys_from_h5(h5_file):
    res = []
    for key in h5_file.keys():
        res.append(key)
    return res

all_file_full_path_list = []
all_file_name_list = []
label_list = []
def get_all_files(path):
    all_file_list = os.listdir(path)
    all_file_list.sort()
    # 遍历该文件夹下的所有目录或文件
    for file in all_file_list:
        file_path = os.path.join(path, file)
        # 如果是文件夹，递归调用当前函数
        if os.path.isdir(file_path):
            get_all_files(file_path)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(file_path) and file.startswith(('.')) == False:
            all_file_full_path_list.append(file_path)
            all_file_name_list.append(file)
            datestr = file.split('_')[2].split('.')[0]
            startTime = datetime.datetime.strptime(datestr, "%Y%m%d%H%M")
            label_list.append(startTime)
    return all_file_full_path_list, all_file_name_list,label_list


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





