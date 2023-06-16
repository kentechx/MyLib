import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


def img_to_u1(img):
    if img.max() <= 1.:
        img = (img * 255).astype('u1')
    else:
        img = img.astype('u1')
    return img


def get_grid_size(n):
    return math.ceil(math.sqrt(n))


def is_onehot(labels):
    return labels.ndim == 4


def get_frame(data, i, axis=0):
    idx = [slice(None)] * data.ndim
    idx[axis] = i
    return data[tuple(idx)]


def get_start_end_frame(labels: np.ndarray, axis=0):
    if is_onehot(labels):
        labels = labels.max(axis=0)
    dims = set(range(labels.ndim)) - {axis}
    labels = labels.max(axis=tuple(dims))
    idx = np.where(labels > 0)[0]
    return idx.min(), idx.max()


class MedicalImagingHelper:

    @staticmethod
    def show_ct_label(data, labels, axis=0, n_images=16):
        n_row = get_grid_size(n_images)
        n_col = n_row
        start_i, end_i = get_start_end_frame(labels, axis)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
        step = (end_i - start_i + n_images - 1) // n_images
        for i, i_frame in enumerate(range(start_i, end_i, step)):
            ax = axes[i // n_col, i % n_col]
            ax.imshow(get_frame(data, i_frame, axis), cmap='gray')
            _labels = get_frame(labels, i_frame, axis)
            ax.imshow(_labels, cmap='jet', alpha=np.where(_labels > 0, 0.3, 0))
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_ct(data, labels=None, axis=0, n_images=16, hu_thr=(0, 2100)):
        # data: (d, h, w)
        # label: (d, h, w)
        n_row = get_grid_size(n_images)
        n_col = n_row
        if exists(labels):
            start_i, end_i = get_start_end_frame(labels, axis)
        else:
            start_i, end_i = 0, data.shape[axis]
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
        step = (end_i - start_i + n_images - 1) // n_images
        for i, i_frame in enumerate(range(start_i, end_i, step)):
            ax = axes[i // n_col, i % n_col]
            ax.imshow(get_frame(data, i_frame, axis), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        if exists(labels):
            MedicalImagingHelper.show_ct_label(data, labels, axis, n_images)
