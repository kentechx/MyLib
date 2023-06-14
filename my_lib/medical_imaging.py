import math
import matplotlib
import matplotlib.pyplot as plt


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


def get_grid_size(n):
    return math.ceil(math.sqrt(n))


class MedicalImagingHelper:

    @staticmethod
    def show_ct_label(data, labels, axis=0, n_images=16):
        n_row = get_grid_size(n_images)
        n_col = n_row
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
        step = data.shape[axis] // n_images
        for i in range(n_images):
            ax = axes[i // n_col, i % n_col]
            ax.imshow(data[i * step], cmap='gray')
            ax.imshow(labels[i * step], cmap='jet', alpha=0.3)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_ct(data, labels=None, axis=0, n_images=16, hu_thr=(0, 2100)):
        # data: (d, h, w)
        # label: (d, h, w)
        n_row = get_grid_size(n_images)
        n_col = n_row
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2, n_row * 2))
        step = data.shape[axis] // n_images
        for i in range(n_images):
            ax = axes[i // n_col, i % n_col]
            ax.imshow(data[i * step], cmap='gray')
            # if exists(labels):
            #     ax.imshow(labels[i * step], cmap='jet', alpha=0.3)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        if exists(labels):
            MedicalImagingHelper.show_ct_label(data, labels, axis, n_images)
