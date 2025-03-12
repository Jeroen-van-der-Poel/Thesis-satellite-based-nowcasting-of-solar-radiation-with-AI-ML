import numpy as np
import utils.metrics as metrics
import plot_images
import matplotlib.pyplot as plt
#from pysteps.visualization import plot_precip_field
import os

 # Analysis the cloud forecastresult
def ResultAnalysis(forecast,observe,imagesavefile,temp,metrics_index):
    # cldmask is 0/1
    # cph is 0/1/2
    # cot,cth and reff are numbers, these three factors can use the same metrics
    if temp == 0:
        csi_4, acc_4, pod_4 = ResultAnalysis_cldmask(forecast,observe, 1)
        plot_images.plot_csi(csi_4,'cldmask_csi', 'cldmask', imagesavefile, 'csi.png')
        plot_images.plot_acc(acc_4, 'cldmask_acc','cldmask', imagesavefile, 'acc.png')
        plot_images.plot_pod(pod_4, 'cldmask_pod','cldmask', imagesavefile, 'pod.png')
    elif temp == 1:
        r2_value, rmse_value, mae_value, rrmse_value = ResultAnalysis_cot(forecast,observe,metrics_index)
        plot_images.plot_r2(r2_value, 'cot_r', 'cot', imagesavefile, 'r.png')
        plot_images.plot_rmse(rmse_value / 100, 'cot_rmse', 'cot', imagesavefile, 'rmse.png')
        plot_images.plot_mae(mae_value / 100, 'cot_mae', 'cot', imagesavefile, 'mae.png')
        plot_images.plot_rrmse(rrmse_value, 'cot_rrmse', 'cot', imagesavefile, 'rrmse.png')
    elif temp == 2:
        csi, acc = ResultAnalysis_cph(forecast,observe, 2)
        plot_images.plot_csi(csi,'cph_csi', 'cph', imagesavefile, 'csi.png')
        plot_images.plot_acc(acc,'cph_acc', 'cph', imagesavefile, 'acc.png')
    elif temp == 3:
        r2_value, rmse_value, mae_value, rrmse_value = ResultAnalysis_cot(forecast,observe,metrics_index)
        plot_images.plot_r2(r2_value, 'cth_r', 'cth', imagesavefile, 'r.png')
        plot_images.plot_rmse(rmse_value, 'cth_rmse', 'cth', imagesavefile, 'rmse.png')
        plot_images.plot_mae(mae_value, 'cth_mae', 'cth', imagesavefile, 'mae.png')
        plot_images.plot_rrmse(rrmse_value, 'cth_rrmse', 'cth', imagesavefile, 'rrmse.png')
    elif temp == 4:
        r2_value, rmse_value, mae_value, rrmse_value = ResultAnalysis_cot(forecast,observe,metrics_index)
        plot_images.plot_r2(r2_value,'reff_r', 'reff', imagesavefile, 'r.png')
        plot_images.plot_rmse(rmse_value/ 100,'reff_rmse', 'reff', imagesavefile, 'rmse.png')
        plot_images.plot_mae(mae_value/ 100,'reff_mae', 'reff', imagesavefile, 'mae.png')
        plot_images.plot_rrmse(rrmse_value[:-1],'reff_rrmse', 'reff', imagesavefile, 'rrmse.png')
    elif temp == 5:
        r2_value, rmse_value, mae_value, rrmse_value = ResultAnalysis_cot(forecast,observe,metrics_index)
        plot_images.plot_r2(r2_value,'sds_r', 'sds', imagesavefile, 'r.png')
        plot_images.plot_rmse(rmse_value,'sds_rmse', 'sds', imagesavefile, 'rmse.png')
        plot_images.plot_mae(mae_value,'sds_mae', 'sds', imagesavefile, 'mae.png')
        plot_images.plot_rrmse(rrmse_value[:],'sds_rrmse', 'sds', imagesavefile, 'rrmse.png')

def ResultAnalysis_cot(forecast,observed,metrics_index):
    r2_value = []
    rmse_value = []
    mae_value = []
    rrmse_value =[]
    for i in range(16):
        r2_value = np.append(r2_value, metrics.R(observed[i, :, :], forecast[i, :, :],metrics_index))
        rmse_value = np.append(rmse_value, metrics.RMSE(observed[i, :, :], forecast[i, :, :],metrics_index))
        mae_value = np.append(mae_value, metrics.MAE(observed[i, :, :], forecast[i, :, :],metrics_index))
        rrmse_value = np.append(rrmse_value, metrics.rRMSE(observed[i, :, :], forecast[i, :, :],metrics_index))

    return r2_value,rmse_value,mae_value,rrmse_value

def ResultAnalysis_cldmask(forecast,observed,index):
    csi_value = []
    acc_value = []
    pod_value = []
    for i in range(16):
        csi_value = np.append(csi_value, metrics.CSI(forecast[i, :, :], observed[i, :, :],index, 0.1))
        acc_value = np.append(acc_value, metrics.ACC(forecast[i, :, :], observed[i, :, :],index, 0.1))
        pod_value = np.append(pod_value, metrics.POD(forecast[i, :, :], observed[i, :, :],index, 0.1))
    return csi_value,acc_value,pod_value

def ResultAnalysis_cph(forecast,observed,index):
    csi_value = []
    acc_value = []
    pod_value = []
    for i in range(16):
        csi_value = np.append(csi_value, metrics.CSI(observed[i, :, :], forecast[i, :, :],index, 0.1))
        acc_value = np.append(acc_value, metrics.ACC(observed[i, :, :], forecast[i, :, :],index, 0.1))
    return csi_value,acc_value

def RadiationAnalysis_metrics(rad_obv_16,sds_acc_crop,sds_inst_crop,label_ins,label_acc,imagesavefile,metrics_output_filename,metrics_index):
    image_file = imagesavefile + metrics_output_filename
    if not os.path.exists(image_file):
      os.makedirs(image_file)
    r2_ins, rmse_ins, mae_ins, rrmse_ins = ResultAnalysis_cot(sds_inst_crop, rad_obv_16,metrics_index)
    plot_images.plot_r2(r2_ins[:-4],label_ins, 'Forecast_radiation', image_file, 'r.png')
    plot_images.plot_rmse(rmse_ins[:-4],label_ins,'Forecast_radiation', image_file, 'rmse.png')
    plot_images.plot_mae(mae_ins[:-4],label_ins, 'Forecast_radiation', image_file, 'mae.png')
    plot_images.plot_rrmse( rrmse_ins[:-4],label_ins,'Forecast_radiation', image_file, 'rrmse.png')

def RadiationForecast_analysis_metrics(rad_obv_16, radiation_forecast, sds_inst_crop,sds_cpp, label_forecast,label_ins, label_cpp, imagesavefile,
                              metrics_output_filename,metrics_index):
    image_file = imagesavefile + metrics_output_filename
    if not os.path.exists(image_file):
      os.makedirs(image_file)
    r2_rf, rmse_rf, mae_rf, rrmse_rf = ResultAnalysis_cot(radiation_forecast, rad_obv_16,metrics_index)
    r2_ins, rmse_ins, mae_ins, rrmse_ins = ResultAnalysis_cot(sds_inst_crop, rad_obv_16,metrics_index)
    r2_cpp, rmse_cpp, mae_cpp, rrmse_cpp = ResultAnalysis_cot(sds_cpp, rad_obv_16,metrics_index)

    plot_images.plot_triple(r2_rf, r2_ins,r2_cpp, label_forecast, label_ins, label_cpp, "R", 'Forecast_radiation', image_file, 'r.png')
    plot_images.plot_triple(rmse_rf, rmse_ins, rmse_cpp,  label_forecast, label_ins, label_cpp,"RMSE(w/m2)", 'Forecast_radiation', image_file, 'rmse.png')
    plot_images.plot_triple(mae_rf, mae_ins, mae_cpp, label_forecast, label_ins, label_cpp, "MAE(w/m2)",'Forecast_radiation', image_file, 'mae.png')
    plot_images.plot_triple(rrmse_rf, rrmse_ins,rrmse_cpp, label_forecast, label_ins, label_cpp, "rRMSE(%)", 'Forecast_radiation', image_file,'rrmse.png')