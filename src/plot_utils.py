"""Plotting, display, and stimulus-loading utilities.

Covers:
  - createFlowDataset — generate optical flow stimulus arrays from PNG files
  - subps      — matplotlib subplot wrapper
  - twx        — twin-x axis helper
  - npprint    — numpy array pretty-print
  - plot_image — display a single image array
  - plot_images — display a grid of image arrays
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def createFlowDataset(categories, topdir, mydirs, orig_shape, input_shape, scl_factor, N_INSTANCES, trial_len, stride):
    """Generate optical flow stimulus arrays from pre-rendered PNG files.

    Args:
        categories: list of stimulus category names (e.g. ['grat_W1', ...])
        topdir: root directory containing ``{category}_inst{i}/{direction}/0.png``
        mydirs: list of motion direction strings (e.g. ['0', '45', ..., '315'])
        orig_shape: (width, height) of source images
        input_shape: (height, width) to crop from centre
        scl_factor: resize scale factor (1 = no resize)
        N_INSTANCES: number of instances per category
        trial_len: number of temporal frames
        stride: pixel shift per frame

    Returns:
        dict mapping instance index → concatenated stimulus array (NDIRS*frames, pixels)
    """
    scld_shape = tuple((np.array(orig_shape) * scl_factor).astype('int'))
    NDIRS = len(mydirs)
    frames_per_stim = int(np.ceil(trial_len / stride))

    shift_foos = {
        '0':   lambda im, step: np.roll(im, step, 1),
        '45':  lambda im, step: np.roll(np.roll(im, step, 1), -step, 0),
        '90':  lambda im, step: np.roll(im, -step, 0),
        '135': lambda im, step: np.roll(np.roll(im, -step, 1), -step, 0),
        '180': lambda im, step: np.roll(im, -step, 1),
        '225': lambda im, step: np.roll(np.roll(im, -step, 1), step, 0),
        '270': lambda im, step: np.roll(im, step, 0),
        '315': lambda im, step: np.roll(np.roll(im, step, 0), step, 1),
    }

    flow_datasets = {}

    for inst_i in range(N_INSTANCES):
        print('*INSTANCE', inst_i, end=' ', flush=True)
        for cat in categories:
            print('.', end='', flush=True)
            stim_arrays = None

            for di, d in enumerate(mydirs):
                image_path = f'{topdir}/{cat}_inst{inst_i}/{d}/0.png'
                img = Image.open(image_path)
                assert orig_shape == img.size

                if scl_factor != 1:
                    img = img.resize(scld_shape, Image.Resampling.LANCZOS)

                w, h = img.size
                assert w == scld_shape[0] and h == scld_shape[1]
                i0 = h // 2 - input_shape[0] // 2
                j0 = w // 2 - input_shape[1] // 2
                i1 = i0 + input_shape[0]
                j1 = j0 + input_shape[1]

                img_array = np.array(img)[:, :, 0]

                for fii, fi in enumerate(range(0, trial_len, stride)):
                    shifted_img = shift_foos[d](img_array, fi)[i0:i1, j0:j1]
                    if stim_arrays is None:
                        stim_arrays = np.zeros((NDIRS * frames_per_stim, shifted_img.size))
                    stim_arrays[di * frames_per_stim + fii] = shifted_img.ravel()

            if inst_i not in flow_datasets:
                flow_datasets[inst_i] = stim_arrays
            else:
                flow_datasets[inst_i] = np.concatenate([flow_datasets[inst_i], stim_arrays])

        print()
    return flow_datasets


def subps(nrows, ncols, rowsz=3, colsz=4, d3=False, axlist=False):
    """Create a matplotlib figure with a grid of subplots.

    Args:
        nrows, ncols: grid dimensions
        rowsz, colsz: per-subplot height and width in inches
        d3: if True, use 3D projection
        axlist: if True and nrows*ncols==1, wrap axes in a list

    Returns:
        (fig, axes)
    """
    if d3:
        f = plt.figure(figsize=(ncols * colsz, nrows * rowsz))
        axes = [
            [f.add_subplot(nrows, ncols, ri * ncols + ci + 1, projection='3d') for ci in range(ncols)]
            for ri in range(nrows)
        ]
        if nrows == 1:
            axes = axes[0]
            if ncols == 1:
                axes = axes[0]
    else:
        f, axes = plt.subplots(nrows, ncols, figsize=(ncols * colsz, nrows * rowsz))
    if axlist and ncols * nrows == 1:
        axes = [axes]
    return f, axes


def twx():
    """Create a twin-x axis pair on a new subplot.

    Returns:
        (ax, ax_twin)
    """
    ax = plt.subplot(111)
    return ax, ax.twinx()


def npprint(a, precision=3):
    """Print a numpy array with fixed precision and suppressed scientific notation."""
    with np.printoptions(precision=precision, suppress=True):
        print(a)


def plot_image(orig_image, fig_sz, ax=None, vmin=None, vmax=None, axis_off=True):
    """Display a single 2D image array.

    Args:
        orig_image: 2D array with values in [0, 1] or [0, 255]
        fig_sz: figure size (used when ax is None)
        ax: existing Axes to draw on (creates a new figure if None)
        vmin, vmax: colormap range
        axis_off: hide axis ticks/labels
    """
    image = orig_image.copy()
    assert image.min() >= 0
    if image.max() <= 1:
        image = (image * 255).astype('int32')
    else:
        image = image.astype('int32')

    if ax is None:
        plt.figure(figsize=(fig_sz, fig_sz))
        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(image, vmin=vmin, vmax=vmax)
        if axis_off:
            ax.axis('off')


def plot_images(images_, fig_sz=2, nrows=None, labels=None, vmin=None, vmax=None):
    """Display a grid of images.

    Args:
        images_: array of 2D images
        fig_sz: per-image figure size
        nrows: number of rows (auto-computed if None)
        labels: optional list of x-axis labels per image
        vmin, vmax: colormap range
    """
    images = images_.copy()
    if nrows is not None:
        ncols = int(np.ceil(len(images) / nrows))
    else:
        nrows = int(np.floor(np.sqrt(len(images))))
        ncols = int(np.ceil(np.sqrt(len(images))))
    if ncols * nrows < len(images):
        nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_sz * ncols, fig_sz * nrows))

    for i, ax in enumerate(axes.ravel()):
        if i >= len(images):
            ax.axis('off')
            continue
        img = images[i]
        ax.set_title('%d' % i, size=11)
        if labels is not None:
            plot_image(img, fig_sz, ax, vmin=vmin, vmax=vmax, axis_off=False)
            ax.set_xlabel(labels[i], size=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            plot_image(img, fig_sz, ax, vmin=vmin, vmax=vmax)

    fig.tight_layout()
    plt.show()
