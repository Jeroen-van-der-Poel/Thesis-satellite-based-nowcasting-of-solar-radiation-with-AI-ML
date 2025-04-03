import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils.losses import Loss_hing_disc, Loss_hing_gen
from model.dgmr import DGMR
from utils.utils import *
import numpy as np
import matplotlib
import datetime
from analyse_results import *
from netCDF4 import Dataset

all_file_full_path_list = []
all_file_name_list = []
label_list = []

def get_all_keys_from_nc(nc_file):
    res = []
    for key in nc_file.variables.keys():
        res.append(key)
    return res

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

def Dataprocess_single(INPUT_PATH,windows,height,width):
    all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)
    data1 = np.vstack((all_file_full_path_list, all_file_name_list))
    data_info = np.vstack((data1, label_list)).T
    label_time = []
    length = len(label_list)
    val = length

    dataset_all = []
    csc_all = []
    for id in range(val):
        print(id)
        if id + windows > val - 1:
            break
        print(label_list[id])
        # in case some data is missing, then skip this window and expand the window
        # make sure there are 15 data in each sample
        interval = label_list[id + windows - 1] - label_list[id]
        exact_end_time = datetime.timedelta(minutes=((windows - 1) * 15))
        if interval != exact_end_time:
            id = id + 1
            print('Sorry, skip lost files')
            continue

        nc_file_start = Dataset(all_file_full_path_list[id], mode='r')
        nc_file_end = Dataset(all_file_full_path_list[id + 3], mode='r')
        start_time = label_list[id]
        end_time = label_list[id + 3]
        if start_time.hour < 5 or start_time.hour > 20:
            continue
        if np.all(nc_file_start['sds'][:] < 0):
            print('start has all -1 ' + str(label_list[id]))
            continue
        if np.all(nc_file_end['sds'][:] < 0):
            print('end has all -1 ' + str(label_list[id]))
            continue
        if np.any(nc_file_start['sds'][:] < 0) and start_time.hour <= 9:
            print('Skip files with input with some -1 ' + str(label_list[id]))
            continue
        sds_csc = np.zeros(shape=(windows, height, width))
        dataset = np.zeros(shape=(windows, height, width))
        for j in range(windows):
            nc_file = Dataset(all_file_full_path_list[id + j], mode ='r')
            keys = get_all_keys_from_nc(nc_file)
            # calculate cth
            #ele = h5_file[keys[3]][:] /15000
            # calculate sds
            # ele = h5_file[keys[7]][:] / 10000
            ele_sds = nc_file['sds'][:]
            ele_sds[ele_sds < 0] = 0
            ele_cs = nc_file['sds_cs'][:]
            ele_cs[ele_cs < 0] = 0
            ele = ele_sds / ele_cs
            ele[np.isnan(ele)] = 0
            # note that here it is (width, heigh) while in the tensor is in (rows = height, cols = width)
            dataset[j] = np.array(ele.T)
            sds_csc[j] = np.array(ele_cs.T)
        label_time.append(label_list[id:id + windows])
        dataset_all.append(dataset)
        csc_all.append(sds_csc)
    dataset_all = np.concatenate([dataset[None,:, :, :] for dataset in dataset_all])
    csc_all = np.concatenate([dataset[None, :, :, :] for dataset in csc_all])
    #label_time = np.concatenate([lab[None, :] for lab in label_time])
    dataset_all = np.expand_dims(dataset_all, axis=-1)
    csc_all = np.expand_dims(csc_all, axis=-1)
    return dataset_all,csc_all, label_time

def Dataprocess_multi_5(INPUT_PATH,model_path,windows,height,width,depth):
    all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)
    data1 = np.vstack((all_file_full_path_list, all_file_name_list))
    data_info = np.vstack((data1, label_list)).T
    label_time = []
    length = len(label_list)
    val = length
    dataset_all = []
    for id in range(val):
        print(id)
        if id + windows > val - 1:
            break
        print(label_list[id])
        # in case some data is missing, then skip this window and expand the window
        # make sure there are 15 data in each sample
        interval = label_list[id + windows - 1] - label_list[id]
        exact_end_time = datetime.timedelta(minutes=((windows - 1) * 15))
        if interval != exact_end_time:
            id = id + 1
            print('Sorry, skip lost files')
            continue
        dataset = np.zeros(shape=(windows, height, width,depth))
        for j in range(windows):
            nc_file = Dataset(all_file_full_path_list[id + j], mode='r')
            keys = get_all_keys_from_nc(nc_file)
            # calculate cth
            #cldmask = np.array(h5_file[keys[0]][:]).T
            #cot = np.array(h5_file[keys[1]][:]).T / 15000
            cth = np.array(nc_file[keys[3]][:]).T / 16537
            reff = np.array(nc_file[keys[6]][:]).T / 6000
            sds = np.array(nc_file[keys[7]][:]).T / 10000
            matrix = np.stack((cth, reff, sds), axis=-1)
            # matrix = np.stack((cldmask, cot, cth, reff, sds), axis=-1)
            # note that here it is (width, heigh) while in the tensor is in (rows = height, cols = width)
            dataset[j] = matrix
        label_time.append(label_list[id:id + windows])
        dataset_all.append(dataset)
    dataset_all = np.concatenate([dataset[None,:, :, :] for dataset in dataset_all])
    #label_time = np.concatenate([lab[None, :] for lab in label_time])
    # dataset_all = np.expand_dims(dataset_all, axis=-1)
    return dataset_all, label_time

def Dataprocess_multi_2(INPUT_PATH,model_path,windows,height,width,depth):
    all_file_full_path_list, all_file_name_list, label_list = get_all_files(INPUT_PATH)
    data1 = np.vstack((all_file_full_path_list, all_file_name_list))
    data_info = np.vstack((data1, label_list)).T
    label_time = []
    length = len(label_list)
    val = length
    dataset_all = []
    for id in range(val):
        print(id)
        if id + windows > val - 1:
            break
        print(label_list[id])
        # in case some data is missing, then skip this window and expand the window
        # make sure there are 15 data in each sample
        interval = label_list[id + windows - 1] - label_list[id]
        exact_end_time = datetime.timedelta(minutes=((windows - 1) * 15))
        if interval != exact_end_time:
            id = id + 1
            print('Sorry, skip lost files')
            continue
        dataset = np.zeros(shape=(windows, height, width,depth))
        for j in range(windows):
            nc_file = Dataset(all_file_full_path_list[id + j], mode='r')
            keys = get_all_keys_from_nc(nc_file)
            # sds
            ele_sds = nc_file[keys[7]][:]
            ele_sds[ele_sds < 0] = 0
            # sds_cs
            # ele_cs = h5_file[keys[8]][:]
            # ele_cs[ele_cs < 0] = 0
            sds = ele_sds / 10000
            sds[np.isnan(sds)] = 0
            sunz = nc_file[keys[10]][:] / 100
            sunz[sunz < 0] = 0
            sunz = np.radians(sunz)
            sunz_cos = np.cos(sunz)
            matrix = np.stack((sds.T, sunz_cos.T), axis=-1)
            # matrix = np.stack((cldmask, cot, cth, reff, sds), axis=-1)
            # note that here it is (width, heigh) while in the tensor is in (rows = height, cols = width)
            dataset[j] = matrix
        label_time.append(label_list[id:id + windows])
        dataset_all.append(dataset)
    dataset_all = np.concatenate([dataset[None,:, :, :] for dataset in dataset_all])
    #label_time = np.concatenate([lab[None, :] for lab in label_time])
    # dataset_all = np.expand_dims(dataset_all, axis=-1)
    return dataset_all, label_time

#seperate all the data into training and validation data,and randomly choose a timeseries for predicting
#this is mainly for test
def PredictData_valid_multi(INPUT_PATH, model_path, windows, height, width,depth):
    dataset, label_time = Dataprocess_multi_5(INPUT_PATH, model_path, windows, height, width,depth)
    label_time = np.array(label_time)
    label_x, labe_y = split_time_xy(label_time)
    predict_x, predict_y = split_data_xy(dataset)
    random_index = np.random.choice(range(len(predict_x)), size=1)
    predict_x = predict_x[random_index[0]]
    predict_y = predict_y[random_index[0]]
    labe_y = labe_y[random_index[0]]

    predict_x = np.expand_dims(predict_x, axis=0)
    predict_x = tf.convert_to_tensor(predict_x,dtype=tf.dtypes.float32)
    #the size of images need to be changed to (224,128), in order to march the model
    predict_x = tf.pad(predict_x, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
    my_model = creatmodel(model_path)
    new_prediction = my_model.generator_obj(predict_x)
    new_prediction = np.array(new_prediction)
    #crop the image into the original size
    new_prediction = new_prediction[:,:,:208,:126,:]

    new_prediction = np.squeeze(new_prediction)
    new_prediction = new_prediction * 10000
    original_frames = predict_y[:, :, :, 2:3]
    original_frames = np.squeeze(original_frames) * 10000
    return new_prediction,original_frames,labe_y

def PredictData_valid_multi_2(INPUT_PATH, model_path, windows, height, width,depth):
    dataset, label_time = Dataprocess_multi_2(INPUT_PATH, model_path, windows, height, width,depth)
    label_time = np.array(label_time)
    label_x, labe_y = split_time_xy(label_time)
    predict_x, predict_y = split_data_xy(dataset)
    random_index = np.random.choice(range(len(predict_x)), size=1)
    random_index[0] = 96
    predict_x = predict_x[random_index[0]]
    predict_y = predict_y[random_index[0]]
    labe_y = labe_y[random_index[0]]

    predict_x = np.expand_dims(predict_x, axis=0)
    predict_x = tf.convert_to_tensor(predict_x[:,:,:,:,0:2],dtype=tf.dtypes.float32)
    #the size of images need to be changed to (224,128), in order to march the model
    predict_x = tf.pad(predict_x, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
    my_model = creatmodel(model_path)
    new_prediction = my_model.generator_obj(predict_x)
    new_prediction = np.array(new_prediction)
    #crop the image into the original size
    new_prediction = new_prediction[:,:,:208,:126,:]

    new_prediction = np.squeeze(new_prediction)
    # cs_frames = predict_y[:, :, :, 2:3]
    # cs_frames = np.squeeze(cs_frames)
    # new_prediction = new_prediction * cs_frames / 10
    new_prediction = new_prediction * 1000

    original_frames = np.squeeze(predict_y[:,:,:,0:1]) * 1000

    return new_prediction,original_frames,labe_y

def ResultOutput(sds,file_out,str_time):
    fname = 'GAN_forecast_'
    starttime = str_time[0] - datetime.timedelta(minutes=15)
    starttime = str(starttime.year) + str(starttime.month).zfill(2) + str(starttime.day).zfill(2) + str(
          starttime.hour).zfill(2) + str(starttime.minute).zfill(2)

    file_out = file_out + starttime[0:8] + '/'
    if not os.path.exists(file_out):
        os.makedirs(file_out)
    for i in range(16):
      forecast_time = str_time[i]
      forecast_time = str(forecast_time.year) + str(forecast_time.month).zfill(2) + str(forecast_time.day).zfill(2) + str(
          forecast_time.hour).zfill(2) + str(forecast_time.minute).zfill(2)
      file_name = fname + starttime + "_F" + forecast_time + '.h5'

      fout = h5py.File(file_out + file_name, 'w')
      fout.create_dataset('sds', data=(sds[i, :, :]).astype('int32'))
      fout.close()
    return

def PredictData_valid(INPUT_PATH, model_path, windows, height, width):
    dataset,sds_csc, label_time = Dataprocess_single(INPUT_PATH, windows, height, width)
    label_time = np.array(label_time)
    label_x, time_y = split_time_xy(label_time)
    image_x, image_y = split_data_xy(dataset)
    csc_x, sds_csc_y = split_data_xy(sds_csc)
    my_model = creatmodel(model_path)

    for i in range(len(dataset)):
        # i = 65 + i
        predict_x = image_x[i]
        predict_y = image_y[i]
        csc_y = sds_csc_y[i]
        labe_y = time_y[i]

        predict_x = np.expand_dims(predict_x, axis=0)
        predict_x = tf.convert_to_tensor(predict_x, dtype=tf.dtypes.float32)
        # the size of images need to be changed to (224,128), in order to match the model
        predict_x = tf.pad(predict_x, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
        print('Predicting ' + str(i) + '/' + str(len(dataset)))
        new_prediction = my_model.generator_obj(predict_x, is_training=True)
        new_prediction = np.array(new_prediction)
        # crop the image into the original size
        new_prediction = new_prediction[:, :, :208, :126, :]
        new_prediction = np.squeeze(new_prediction)
        csc_y = np.squeeze(csc_y)

        new_prediction = new_prediction * csc_y / 10
        original_frames = np.squeeze(predict_y) * csc_y / 10

        new_prediction[new_prediction < 0] = 0
        new_prediction[new_prediction > 1000] = 1000

        if i < 300:
            # if i == 65:
            #     a = 0
            filename = str(labe_y[0]).replace('-', '').replace(' ', '').replace(':', '')
            image_path_result = './Evaluation_image/' + filename + '.png'
            figshow(original_frames, new_prediction, labe_y, image_path_result)

            image_path_matric = './Evaluation-metrics/' + filename + '/'
            if not os.path.exists(image_path_matric):
                os.makedirs(image_path_matric)
            ResultAnalysis(new_prediction, original_frames, image_path_matric, 5, 0)

        ResultOutput(new_prediction, file_out, labe_y)
    return 1


def figshow(original_frames,new_prediction,label_y,imagesavefile):
    original_frames = np.squeeze(original_frames)
    new_prediction = np.squeeze(new_prediction)
    fig, axes = plt.subplots(4, 8, figsize=(20, 4), sharex=True, sharey=True)
    datestr = label_y[0].strftime("%Y-%m-%d")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)
    fig.suptitle('sds ' + datestr)
    # Plot the original frames.
    for idx, ax in enumerate(axes[0]):
        ax.imshow((original_frames[idx]).T, cmap="viridis",norm=norm)
        ax.set_title(label_y[idx].strftime("%H:%M:%S"))
        ax.axis("off")
    for idx, ax in enumerate(axes[1]):
        ax.imshow((original_frames[idx + 8]).T, cmap="viridis",norm=norm)
        ax.set_title(label_y[idx + 8].strftime("%H:%M:%S"))
        ax.axis("off")
    # Plot the predicted frames.
    for idx, ax in enumerate(axes[2]):
        ax.imshow((new_prediction[idx]).T, cmap="viridis",norm=norm)
        ax.set_title(label_y[idx].strftime("%H:%M:%S"))
        ax.axis("off")
    for idx, ax in enumerate(axes[3]):
        im = ax.imshow((new_prediction[idx + 8]).T, cmap="viridis",norm=norm)
        ax.set_title(label_y[idx + 8].strftime("%H:%M:%S"))
        ax.axis("off")

    fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, bottom=0.1)
    position = fig.add_axes([0.92, 0.12, 0.015, .78])
    cbar = fig.colorbar(im, cax = position,ax=axes[-1,:])
    plt.savefig(os.path.join(imagesavefile))


# initialize model
def creatmodel(CHECKPOINT_DIR):
    disc_optimizer = Adam(learning_rate=2E-4, beta_1=0.0, beta_2=0.999)
    gen_optimizer = Adam(learning_rate=1E-5, beta_1=0.0, beta_2=0.999)
    loss_hinge_gen = Loss_hing_gen()
    loss_hinge_disc = Loss_hing_disc()
    my_model = DGMR(lead_time=240, time_delta=15)
    my_model.compile(gen_optimizer, disc_optimizer,
                     loss_hinge_gen, loss_hinge_disc)
    # load weights
    ckpt = tf.train.Checkpoint(generator=my_model.generator_obj,
                               discriminator=my_model.discriminator_obj,
                               generator_optimizer=my_model.gen_optimizer,
                               discriminator_optimizer=my_model.disc_optimizer)

    # ckpt.restore(ROOT /'Checkpoints/sds_single_day_v0801_y10/ckpt-205.index')

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, CHECKPOINT_DIR, max_to_keep=10)
    if ckpt_manager.latest_checkpoint:
        # ckpt.restore(ckpt_manager.latest_checkpoint)
        ckpt.restore(ckpt_manager.checkpoints[-34])
        print('Latest checkpoint restored!!')
    return my_model

if __name__ == "__main__":
    # read path to checkpoint
    # INPUT_PATH = '/nobackup/users/cui/2022-MSGCPP_CSC/04/20220407/'
    INPUT_PATH = '/nobackup/users/cui/2022-MSGCPP_CSC/01/'
    file_out = '/nobackup/users/cui/CPPv2_inout/CPPout/2022/01-GAN/'
    windows = 20
    height = 208
    width = 126

    ROOT = get_project_root()
    #model_path = ROOT / 'Checkpoints/sds_single_V0608_v0/160000/'
    model_path = ROOT /'Checkpoints/0925_V03/'
    #model_path = ROOT /'Checkpoints/cth-0526/310000'
    # PredictData_valid(INPUT_PATH,model_path, windows, height, width)
    PredictData_valid(INPUT_PATH,model_path, windows, height, width)
    # filename = str(label_y[0]).replace('-','').replace(' ','').replace(':','')
    # image_path_result = './Evaluation_image/' + filename + '.png'
    # filename = str(label_y[0])
    # figshow(original_frames, new_prediction, label_y, image_path_result)
    '''
    #-----------------------------------------------------------------------------------------------
    # These indexes below are for analysis result output
    # change the number in [-1,5] to choose which image you want to print
    # temp is used to control the indexes
    # no images printed = -1
    # cldmask_ = 0
    # cot_ = 1
    # cph = 2
    # cth = 3
    # reff = 4
    # sds = 5
    #The evaluation metrics are used for cloud and radiation, they have different masks
    #Use metrics_index to indicate different objects
    #metrics_index = 0 cloud parameters
    #metrics_index = 1 radiation parameters
    '''
    # image_path_matric = './Evaluation-metrics/'
    # ResultAnalysis(new_prediction,original_frames,image_path_matric,5,0)
