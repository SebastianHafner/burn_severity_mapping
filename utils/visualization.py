import matplotlib.colors as colors
import numpy as np


def plot_s2(cfg, ax, img: np.ndarray, vis_bands: tuple = ('B11', 'B8', 'B12'), scale_factor: float = 0.4):
    available_bands = cfg.DATASET.S2_BANDS
    vis_selection = [available_bands.index(band) for band in vis_bands]
    img = np.clip(img[:, :, vis_selection] / scale_factor, 0, 1)
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_burn_severity(cfg, ax, bs_map: np.ndarray):
    n_classes = len(cfg.DATASET.CLASSES)
    colormap = colors.ListedColormap(cfg.DATASET.COLORS)
    boundaries = np.arange(-0.5, n_classes)
    norm = colors.BoundaryNorm(boundaries, colormap.N, clip=True)
    ax.imshow(bs_map, cmap=colormap, norm=norm, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])