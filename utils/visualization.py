import matplotlib.colors as colors
import numpy as np

# TODO: add visualization functions for worldview image and sentinel-2 image


def plot_classification(cfg, ax, classification: np.ndarray, title: str = None):

    n_classes = cfg.MODEL.OUT_CHANNELS
    cmap = colors.ListedColormap(cfg.DATASET.LABEL.REMAPPER.REMAPPED_COLORS)

    boundaries = np.arange(-0.5, n_classes)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(classification, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_probability(ax, prob: np.ndarray, title: str = None):
    ax.imshow(prob, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_original_classes(cfg, ax, landcover: np.ndarray, title: str = None):
    n_classes = len(cfg.DATASET.LABEL.CLASSES)
    cmap = colors.ListedColormap(cfg.DATASET.LABEL.CLASS_COLORS)
    boundaries = np.arange(-0.5, n_classes)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(landcover, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_s2(cfg, ax, raw_img: np.ndarray, vis_bands: tuple = ('B4', 'B3', 'B2'), title: str = None,
            scale_factor: float = 0.4):
    available_bands = cfg.DATASET.SATELLITE.AVAILABLE_S2_BANDS
    vis_selection = [available_bands.index(band) for band in vis_bands]
    img = np.clip(raw_img[:, :, vis_selection] / 10_000 / scale_factor, 0, 1)
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_wv(cfg, ax, raw_img: np.ndarray, vis_bands: list = ('b5', 'b3', 'b2'), title: str = None,
            scale_factor: float = 0.5):
    available_bands = cfg.DATASET.SATELLITE.AVAILABLE_WV_BANDS
    vis_selection = [available_bands.index(band) for band in vis_bands]
    img = np.clip(raw_img[:, :, vis_selection] / 1_000 / scale_factor, 0, 1)
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
