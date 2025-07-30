from pathlib import Path
import yaml
import os
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import os

def get_project_root() -> Path:
    return Path(__file__).parent

def read_yaml(file_path) -> str:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def make_dirs(list_dir: list) -> None:
    for l in list_dir:
        l.mkdir(parents=True, exist_ok=True)

def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size >> 20


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def crop_middle(tensor):
    # Prepare the frames for temporal discriminator: choose the offset of a
    # random crop of size 128x128 out of 256x256 and pick full sequence samples.
    b, t, h, w, c = tensor.shape
    cr = 2
    h_offset = (h // cr) // 2
    w_offset = (w // cr) // 2
    zero_offset = tf.zeros_like(w_offset)
    begin_tensor = tf.stack(
        [zero_offset, zero_offset, h_offset, w_offset, zero_offset], -1)
    size_tensor = tf.constant([b, t, h // cr, w // cr, c])
    frames_for_eval = tf.slice(tensor, begin_tensor, size_tensor)
    frames_for_eval.set_shape([b, t, h // cr, w // cr, c])
    return frames_for_eval

def crop_middle_ensemble(tensor):
    # Prepare the frames for temporal discriminator: choose the offset of a
    b, e, t, h, w, c = tensor.shape
    cr = 2
    h_offset = (h // cr) // 2
    w_offset = (w // cr) // 2
    zero_offset = tf.zeros_like(w_offset)
    begin_tensor = tf.stack(
        [zero_offset, zero_offset, zero_offset, h_offset, w_offset, zero_offset], -1)
    size_tensor = tf.constant([b, e, t, h // cr, w // cr, c])
    frames = tf.slice(tensor, begin_tensor, size_tensor)
    frames.set_shape([b, e, t, h // cr, w // cr, c])
    return frames