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

def R2(obs, sim):
    obs = obs.flatten()
    sim = sim.flatten()

    numerator = ((obs - sim) ** 2).sum()
    denominator = ((obs - np.mean(obs)) ** 2).sum()
    return 1 - numerator/denominator

def r2Analysis(sds,cloud):
    r2_value = []
    for i in range(16):
        r2_value = R2(sds, cloud)
    return r2_value

def rAnalysis(sds,cloud):
    obs = sds.flatten()
    sim = cloud.flatten()

    return np.corrcoef(obs, sim)[0, 1]

all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)

length = len(label_list)
r_cldmask = []
r_cth = []
r_cph = []
r_cot = []
r_reff = []
for i in range(length):
    h5_file = h5py.File(all_file_full_path_list[i])
    keys = get_all_keys_from_h5(h5_file)
    sds = h5_file[keys[7]][:]
    for j in range(7):
        if j == 0:
            ele = h5_file[keys[j]][:]
            #normalization
            r = rAnalysis(sds/np.max(sds), ele/np.max(ele))
            if math.isinf(r) or math.isnan(r):
               continue
            else:
               r_cldmask.append(r)
        elif j == 1:
            ele = h5_file[keys[j]][:]
            # normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_cot.append(r)
        elif j == 2:
            ele = h5_file[keys[j]][:]
            # normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_cph.append(r)
        elif j == 3:
            ele = h5_file[keys[j]][:]
            # normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_cth.append(r)
        elif j == 6:
            ele = h5_file[keys[j]][:]
            # normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_reff.append(r)
        else:
            continue
#image output
fig, ax = plt.subplots() # 创建图实例
x = np.arange(len(r_cldmask)) # 创建x的取值范围

plt.plot(x,r_cldmask,label='cldmask')
plt.plot(x,r_cth,label='cth')
plt.plot(x,r_cph,label='cph')
plt.plot(x,r_reff,label='reff')
plt.plot(x,r_cot,label='cot')
plt.legend()
plt.show()
a = 0





