import math
import h5py
import matplotlib.pyplot as plt
import numpy as np
from data_preperation_to_tfr import get_all_keys_from_h5, get_all_files

INPUT_PATH = './202112/'

all_file_full_path_list = []
all_file_name_list = []
label_list = []

# Calculate the coefficient of determination
def R2(obs, sim):
    obs = obs.flatten()
    sim = sim.flatten()

    numerator = ((obs - sim) ** 2).sum()
    denominator = ((obs - np.mean(obs)) ** 2).sum()
    return 1 - numerator/denominator

# Compute the Pearson correlation coefficient
def rAnalysis(sds,cloud):
    obs = sds.flatten()
    sim = cloud.flatten()

    return np.corrcoef(obs, sim)[0, 1]

all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)

length = len(label_list)
r_cldmask = []      # cloud mask correlation with SDS
r_cth = []          # cloud top height correlation
r_cph = []          # cloud phase correlation
r_cot = []          # cloud optical thickness correlation
r_reff = []         # effective cloud particle radius correlation

for i in range(length):
    h5_file = h5py.File(all_file_full_path_list[i])
    keys = get_all_keys_from_h5(h5_file)
    sds = h5_file[keys[7]][:]
    for j in range(7):
        if j == 0:
            ele = h5_file[keys[j]][:]
            # Normalization
            r = rAnalysis(sds/np.max(sds), ele/np.max(ele))
            if math.isinf(r) or math.isnan(r):
               continue
            else:
               r_cldmask.append(r)
        elif j == 1:
            ele = h5_file[keys[j]][:]
            # Normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_cot.append(r)
        elif j == 2:
            ele = h5_file[keys[j]][:]
            # Normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_cph.append(r)
        elif j == 3:
            ele = h5_file[keys[j]][:]
            # Normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_cth.append(r)
        elif j == 6:
            ele = h5_file[keys[j]][:]
            # Normalization
            r = rAnalysis(sds / np.max(sds), ele / np.max(ele))
            if math.isinf(r) or math.isnan(r):
                continue
            else:
                r_reff.append(r)
        else:
            continue

# Image output
fig, ax = plt.subplots()
x = np.arange(len(r_cldmask))
plt.plot(x,r_cldmask,label='cldmask')
plt.plot(x,r_cth,label='cth')
plt.plot(x,r_cph,label='cph')
plt.plot(x,r_reff,label='reff')
plt.plot(x,r_cot,label='cot')
plt.legend()
plt.show()
a = 0