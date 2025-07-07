# EarthFormer/visualization/sevir/sevir_vis_seq_denom.py

import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from EarthFormer.utils.layout import change_layout_np
from .sevir_cmap import get_cmap

# === CONFIG ===
SSI_VMIN = 0.0
SSI_VMAX = 1000.0  # Adjust if your irradiance exceeds this
THRESHOLDS = (0, 100, 300, 500, 600, 700, 850, 1000)

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255


def cmap_dict_auto(data):
    return {
        'cmap': jet_with_gray(),
        'norm': Normalize(vmin=SSI_VMIN, vmax=SSI_VMAX),
    }


def jet_with_gray():
    base = plt.cm.get_cmap('jet', 256)
    base_colors = base(np.linspace(0, 1, 256))
    base_colors[0] = [0.5, 0.5, 0.5, 1]
    return LinearSegmentedColormap.from_list('jet_gray', base_colors)


def plot_hit_miss_fa(ax, y_true, y_pred, thres):
    mask = np.zeros_like(y_true)
    mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4  # hit
    mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3   # miss
    mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2   # false alarm
    mask[np.logical_and(y_true < thres, y_pred < thres)] = 1    # correct rejection
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(mask, cmap=cmap)


def plot_hit_miss_fa_all_thresholds(ax, y_true, y_pred):
    fig = np.zeros(y_true.shape)
    y_true_idx = np.searchsorted(THRESHOLDS, y_true)
    y_pred_idx = np.searchsorted(THRESHOLDS, y_pred)
    fig[y_true_idx == y_pred_idx] = 4
    fig[y_true_idx > y_pred_idx] = 3
    fig[y_true_idx < y_pred_idx] = 2
    fig[np.logical_and(y_true < THRESHOLDS[1], y_pred < THRESHOLDS[1])] = 1
    cmap = ListedColormap(HMF_COLORS)
    ax.imshow(fig, cmap=cmap)


def visualize_result_horizontal(in_seq, target_seq, pred_seq_list: List[np.array], label_list: List[str],
                     interval_real_time=10.0, idx=0, plot_stride=2, figsize=(8, 24), fs=10,
                     vis_thresh=THRESHOLDS[2], vis_hits_misses_fas=True):
    in_len = in_seq.shape[-1]
    out_len = target_seq.shape[-1]
    max_len = max(in_len, out_len)
    ncols = (max_len - 1) // plot_stride + 1
    if vis_hits_misses_fas:
        fig, ax = plt.subplots(nrows=2 + 3 * len(pred_seq_list), ncols=ncols, figsize=figsize)
    else:
        fig, ax = plt.subplots(nrows=2 + len(pred_seq_list), ncols=ncols, figsize=figsize)

    ax[0][0].set_ylabel('Inputs', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < in_len:
            ax[0][i // plot_stride].imshow(in_seq[idx, :, :, i], **cmap_dict_auto(in_seq[idx, :, :, i]), aspect='auto')
        else:
            ax[0][i // plot_stride].axis('off')

    ax[1][0].set_ylabel('Target', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < out_len:
            ax[1][i // plot_stride].imshow(target_seq[idx, :, :, i], **cmap_dict_auto(target_seq[idx, :, :, i]), aspect='auto')
        else:
            ax[1][i // plot_stride].axis('off')

    target_seq = target_seq[idx:idx + 1]
    y_preds = [pred_seq[idx:idx + 1] for pred_seq in pred_seq_list]

    if vis_hits_misses_fas:
        for k in range(len(pred_seq_list)):
            for i in range(0, max_len, plot_stride):
                if i < out_len:
                    ax[2 + 3 * k][i // plot_stride].imshow(y_preds[k][0, :, :, i], **cmap_dict_auto(y_preds[k][0, :, :, i]), aspect='auto')
                    plot_hit_miss_fa(ax[2 + 1 + 3 * k][i // plot_stride], target_seq[0, :, :, i], y_preds[k][0, :, :, i], vis_thresh)
                    plot_hit_miss_fa_all_thresholds(ax[2 + 2 + 3 * k][i // plot_stride], target_seq[0, :, :, i], y_preds[k][0, :, :, i])
                else:
                    ax[2 + 3 * k][i // plot_stride].axis('off')
                    ax[2 + 1 + 3 * k][i // plot_stride].axis('off')
                    ax[2 + 2 + 3 * k][i // plot_stride].axis('off')

            ax[2 + 3 * k][0].set_ylabel(label_list[k] + '\nPrediction', fontsize=fs)
            ax[2 + 1 + 3 * k][0].set_ylabel(label_list[k] + f'\nScores\nThresh={vis_thresh}', fontsize=fs)
            ax[2 + 2 + 3 * k][0].set_ylabel(label_list[k] + '\nScores\nAll Thresh', fontsize=fs)
    else:
        for k in range(len(pred_seq_list)):
            for i in range(0, max_len, plot_stride):
                if i < out_len:
                    ax[2 + k][i // plot_stride].imshow(y_preds[k][0, :, :, i], **cmap_dict_auto(y_preds[k][0, :, :, i]), aspect='auto')
                else:
                    ax[2 + k][i // plot_stride].axis('off')
            ax[2 + k][0].set_ylabel(label_list[k] + '\nPrediction', fontsize=fs)

    for i in range(0, max_len, plot_stride):
        if i < out_len:
            ax[-1][i // plot_stride].set_title(f'{int(interval_real_time * (i + plot_stride))} Minutes', y=-0.25)

    for row_axes in ax:
        for a in row_axes:
            a.xaxis.set_ticks([])
            a.yaxis.set_ticks([])

    plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.20)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.15, 0.10, 0.7, 0.02])
    cb = plt.colorbar(ScalarMappable(norm=Normalize(vmin=SSI_VMIN, vmax=SSI_VMAX),
                                     cmap=jet_with_gray()),
                      cax=cbar_ax,
                      orientation='horizontal')
    cb.set_label('SSI Intensity (W/m²)', fontsize=12)
    cb.ax.tick_params(labelsize=10)

    if vis_hits_misses_fas:
        legend_elements = [
            Patch(facecolor=HMF_COLORS[3], edgecolor='k', label='Hit'),
            Patch(facecolor=HMF_COLORS[2], edgecolor='k', label='Miss'),
            Patch(facecolor=HMF_COLORS[1], edgecolor='k', label='False Alarm')
        ]
        ax[3][0].legend(handles=legend_elements, loc='center left',
                        bbox_to_anchor=(-2.2, -0.),
                        borderaxespad=0, frameon=False, fontsize='16')

    return fig, ax

def visualize_result_vertical(in_seq, target_seq, pred_seq_list: List[np.array], label_list: List[str],
                              interval_real_time=10.0, idx=0, plot_stride=2, fs=20,
                              vis_thresh=THRESHOLDS[2], vis_hits_misses_fas=True):
    in_len = in_seq.shape[-1]
    out_len = target_seq.shape[-1]
    max_len = max(in_len, out_len)
    nrows = (max_len - 1) // plot_stride + 1

    if vis_hits_misses_fas:
        ncols = 2 + 3 * len(pred_seq_list)
    else:
        ncols = 2 + len(pred_seq_list)

    # === Figure size scaling ===
    fig_width_per_col = 3.0
    fig_height_per_row = 2.5
    figsize = (fig_width_per_col * ncols, fig_height_per_row * nrows)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        ax = [ax]

    for i in range(0, max_len, plot_stride):
        row = i // plot_stride

        if i < in_len:
            ax[row][0].imshow(in_seq[idx, :, :, i], **cmap_dict_auto(in_seq[idx, :, :, i]))
        else:
            ax[row][0].axis('off')

        if i < out_len:
            ax[row][1].imshow(target_seq[idx, :, :, i], **cmap_dict_auto(target_seq[idx, :, :, i]))
        else:
            ax[row][1].axis('off')

        y_preds = [pred_seq[idx:idx + 1] for pred_seq in pred_seq_list]
        for k in range(len(pred_seq_list)):
            if i < out_len:
                ax[row][2 + k].imshow(y_preds[k][0, :, :, i], **cmap_dict_auto(y_preds[k][0, :, :, i]))
            else:
                ax[row][2 + k].axis('off')

        # Left-hand time labels
        # ax[row][0].set_ylabel(f'{int(interval_real_time * (i + plot_stride))} Min', fontsize=fs)
        ax[row][-1].set_ylabel(f'{int(interval_real_time * (i + plot_stride))} Min', fontsize=fs)



    # Column titles
    col_labels = ['Input', 'Target'] + [f'{lbl}\nPred' for lbl in label_list]
    for col, label in enumerate(col_labels):
        ax[0][col].set_title(label, fontsize=fs)

    # Clean ticks
    for row_axes in ax:
        for a in row_axes:
            a.xaxis.set_ticks([])
            a.yaxis.set_ticks([])

    # Adjust layout and add horizontal colorbar
    plt.subplots_adjust(hspace=0.1, wspace=0.05, top=0.95, bottom=0.05)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cb = plt.colorbar(ScalarMappable(norm=Normalize(vmin=SSI_VMIN, vmax=SSI_VMAX),
                                     cmap=jet_with_gray()), cax=cbar_ax)
    cb.set_label('SSI Intensity (W/m²)', fontsize=12)
    cb.ax.tick_params(labelsize=10)

    return fig, ax


def save_example_vis_results(save_dir, save_prefix, in_seq, target_seq, pred_seq, label,
                              layout='NHWT', interval_real_time=10.0, idx=0,
                              plot_stride=2, fs=15, norm=None, vis_hits_misses_fas=True):
    in_seq = change_layout_np(in_seq, in_layout=layout).astype(np.float32)
    target_seq = change_layout_np(target_seq, in_layout=layout).astype(np.float32)
    pred_seq = change_layout_np(pred_seq, in_layout=layout).astype(np.float32)

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f'{save_prefix}.png')

    fig, ax = visualize_result_vertical(
        in_seq=in_seq,
        target_seq=target_seq,
        pred_seq_list=[pred_seq],
        label_list=[label],
        interval_real_time=interval_real_time,
        idx=idx,
        plot_stride=plot_stride,
        fs=fs,
        vis_hits_misses_fas=vis_hits_misses_fas
    )
    plt.savefig(fig_path)
    plt.close(fig)
    plt.close('all')

def save_comparison_vis_results(
        save_dir, save_prefix,
        in_seq, target_seq, pred_seq_list, label_list,
        layout='NHWT', interval_real_time: float = 10.0, idx=0,
        plot_stride=2, fs=14, norm=None, vis_hits_misses_fas=False):
    """
    Save visualization of multiple models compared side-by-side.
    
    Parameters
    ----------
    in_seq : np.array (e.g., NHWT layout)
    target_seq : np.array (NHWT)
    pred_seq_list : List[np.array]
        List of prediction arrays for each model
    label_list : List[str]
        Corresponding labels (e.g., ["EarthFormer", "DGMR-SO", "Persistence"])
    """
    os.makedirs(save_dir, exist_ok=True)

    # Ensure shape layout
    in_seq = change_layout_np(in_seq, in_layout=layout).astype(np.float32)
    target_seq = change_layout_np(target_seq, in_layout=layout).astype(np.float32)
    pred_seq_list = [change_layout_np(pred, in_layout=layout).astype(np.float32) for pred in pred_seq_list]

    in_len = in_seq.shape[-1]
    out_len = target_seq.shape[-1]
    max_len = max(in_len, out_len)
    ncols = (max_len - 1) // plot_stride + 1
    nrows = 2 + len(pred_seq_list)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 2.5 * nrows))

    ax[0][0].set_ylabel('Inputs', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < in_len:
            xt = in_seq[idx, :, :, i]
            ax[0][i // plot_stride].imshow(xt, **cmap_dict_auto(xt))
        else:
            ax[0][i // plot_stride].axis('off')

    ax[1][0].set_ylabel('Target', fontsize=fs)
    for i in range(0, max_len, plot_stride):
        if i < out_len:
            xt = target_seq[idx, :, :, i]
            ax[1][i // plot_stride].imshow(xt, **cmap_dict_auto(xt))
        else:
            ax[1][i // plot_stride].axis('off')

    # Plot each model's prediction
    for k, pred_seq in enumerate(pred_seq_list):
        y_pred = pred_seq[idx:idx + 1]  # single example
        row = 2 + k
        for i in range(0, max_len, plot_stride):
            if i < out_len:
                xt = y_pred[0, :, :, i]
                ax[row][i // plot_stride].imshow(xt, **cmap_dict_auto(xt))
            else:
                ax[row][i // plot_stride].axis('off')
        ax[row][0].set_ylabel(f'{label_list[k]}\nPrediction', fontsize=fs)

    # Time annotations
    for i in range(0, max_len, plot_stride):
        if i < out_len:
            ax[-1][i // plot_stride].set_title(f'{int(interval_real_time * (i + plot_stride))} Minutes', y=-0.25)

    # Clean axes
    for row_axes in ax:
        for a in row_axes:
            a.xaxis.set_ticks([])
            a.yaxis.set_ticks([])

    plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.20)

    # Add shared colorbar
    custom_cmap = jet_with_gray()
    norm = Normalize(vmin=0.0, vmax=1.0)
    cbar_ax = fig.add_axes([0.15, 0.10, 0.7, 0.02])
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=custom_cmap), cax=cbar_ax, orientation='horizontal')
    cb.set_label('SSI Intensity (W/m²)', fontsize=12)
    cb.ax.tick_params(labelsize=10)

    fig_path = os.path.join(save_dir, f'{save_prefix}.png')
    plt.savefig(fig_path)
    plt.close(fig)
    plt.close('all')
