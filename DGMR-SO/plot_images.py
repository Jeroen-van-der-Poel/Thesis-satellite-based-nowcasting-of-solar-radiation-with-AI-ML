import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MultipleLocator

def plot_csi(csi_4,label,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, csi_4, 'r--', label=label)
    #l2 = plt.plot(x, csi_8, 'g--', label='past 8')
    #plt.plot(x, csi_4, 'ro-', x, csi_8, 'g+-')
    plt.xticks((x[::2]),[str(i) for i in (x[::2])])
    plt.xlim(0,255)
    plt.plot(x, csi_4, 'ro-')
    plt.title(index + '-CSI')
    plt.xlabel('minutes')
    plt.ylabel('CSI')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-'+ imagename))
    plt.show()
    plt.close()

def plot_acc(csi_4,label,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, csi_4, 'r--', label=label)
    #l2 = plt.plot(x, csi_8, 'g--', label='past 8')
    #plt.plot(x, csi_4, 'ro-', x, csi_8, 'g+-')
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, csi_4, 'ro-')
    x_major_locator=MultipleLocator(1)
    plt.title(index+'-ACC')
    plt.xlabel('minutes')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-'+ imagename))
    plt.show()
    plt.close()

def plot_pod(csi_4,label,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, csi_4, 'r--', label=label)
    #l2 = plt.plot(x, csi_8, 'g--', label='past 8')
    #plt.plot(x, csi_4, 'ro-', x, csi_8, 'g+-')
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, csi_4, 'ro-')
    plt.title(index+'-POD')
    plt.xlabel('minutes')
    plt.ylabel('POD')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-'+ imagename))
    plt.show()
    plt.close()

def plot_r2(value,label, index,imagesavefile,imagename):
    # plot result
    x = np.arange(15, 255, 15)
    l1 = plt.plot(x, value, 'r--', label=label)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value, 'ro-')
    plt.title(index + '-R')
    plt.xlabel('minutes')
    plt.ylabel('R')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()


def plot_r2_double(value1, value2,label_ins,label_acc, index, imagesavefile, imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, value1, 'r--', label=label_ins)
    l2 = plt.plot(x, value2, 'g--', label=label_acc)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value1, 'ro-', x, value2, 'g+-')
    plt.title(index + '-R2')
    plt.xlabel('minutes')
    plt.ylabel('R2')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()

def plot_rmse_double(value1, value2,label_ins,label_acc,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, value1, 'r--', label=label_ins)
    l2 = plt.plot(x, value2, 'g--', label=label_acc)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value1, 'ro-', x, value2, 'g+-')
    plt.title(index + '-RMSE')
    plt.xlabel('minutes')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()

def plot_mae_double(value1, value2,label_ins,label_acc,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15, 255, 15)
    l1 = plt.plot(x, value1, 'r--', label=label_ins)
    l2 = plt.plot(x, value2, 'g--', label=label_acc)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value1, 'ro-', x, value2, 'g+-')
    plt.title(index + '-MAE')
    plt.xlabel('minutes')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()

def plot_rrmse_double(value1, value2,label_ins,label_acc,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, value1, 'r--', label=label_ins)
    l2 = plt.plot(x, value2, 'g--', label=label_acc)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value1, 'ro-', x, value2, 'g+-')
    plt.title(index + '-rRMSE')
    plt.xlabel('minutes')
    plt.ylabel('rRMSE(%)')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()

def plot_rmse(value,label,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, value, 'r--', label=label)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value, 'ro-')
    plt.title(index + '-RMSE')
    plt.xlabel('minutes')
    plt.ylabel('RMSE(w/m2)')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()

def plot_mae(value,label,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, value, 'r--', label=label)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value, 'ro-')
    plt.title(index + '-MAE')
    plt.xlabel('minutes')
    #plt.ylabel('MAE('+ '$\mu$'+ 'm)')
    plt.ylabel('MAE(w/m2)')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()

def plot_rrmse(value,label,index,imagesavefile,imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, value, 'r--', label=label)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value, 'ro-')
    plt.title(index + '-rRMSE')
    plt.xlabel('minutes')
    plt.ylabel('rRMSE(%)')
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()

def plot_triple(value1, value2,value3, label_1,label_2, label_3, metric_name, index, imagesavefile, imagename):
    # plot result
    x = np.arange(15,255,15)
    l1 = plt.plot(x, value1, 'r--', label=label_1)
    l2 = plt.plot(x, value2, 'g--', label=label_2)
    l3 = plt.plot(x, value3, 'b--', label=label_3)
    plt.xticks((x[::2]), [str(i) for i in (x[::2])])
    plt.plot(x, value1, 'ro-', x, value2, 'g+-',x, value3, 'b+-')
    plt.title(index + '-' + metric_name)
    plt.xlabel('minutes')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(imagesavefile, index + '-' + imagename))
    plt.show()
    plt.close()