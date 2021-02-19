import torch
from torch.utils import data as torch_data
from torchvision import transforms
from abc import abstractmethod
import numpy as np
from utils.augmentations import *
from utils.geofiles import *
import cv2


class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.DATASET.PATH)

        # feature selection
        self.s1_feature_selection = self.get_s1_feature_selection()
        self.s2_feature_selection = self.get_s2_feature_selection()

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    def get_img(self, site: str, x: int, y: int) -> np.ndarray:
        mode = self.cfg.DATASET.MODE
        if mode == 'optical':
            img, _, _ = self.get_s2_data(site, x, y)
        elif mode == 'sar':
            img, _, _ = self.get_s1_data(site, x, y)
        else:
            s1_img, _, _ = self.get_s1_data(site, x, y)
            s2_img, _, _ = self.get_s2_data(site, x, y)
            img = np.concatenate((s1_img, s2_img), axis=-1)
        return img

    def get_s2_data(self, site: str, x: int, y: int, only_postfire: bool = False) -> tuple:
        lvl = self.cfg.DATASET.S2_PROCESSING_LEVEL
        file = self.root_path / site / f's2{lvl}' / f'{site}_s2{lvl}_{y:010d}-{x:010d}.tif'
        img, geotransform, crs = read_tif(file)
        img = img[:, :, self.s2_feature_selection]

        if self.cfg.DATASET.INCLUDE_PREFIRE_S2 and not only_postfire:
            file_prefire = self.root_path / site / f's2{lvl}_prefire' / f'{site}_s2{lvl}_prefire_{y:010d}-{x:010d}.tif'
            img_prefire, geotransform, crs = read_tif(file_prefire)
            img_prefire = img_prefire[:, :, self.s2_feature_selection]
            img = np.concatenate((img_prefire, img), axis=-1)

        return img.astype(np.float32), geotransform, crs

    def get_s1_data(self, site: str, x: int, y: int) -> tuple:
        file = self.root_path / site / f's1' / f'{site}_s1_{y:010d}-{x:010d}.tif'
        img, geotransform, crs = read_tif(file)
        img = img[:, :, self.s1_feature_selection]
        return img.astype(np.float32), geotransform, crs

    def get_firemask(self, site: str, x: int, y: int) -> np.ndarray:
        file = self.root_path / site / 'firemask' / f'{site}_firemask_{y:010d}-{x:010d}.tif'
        mask, geotransform, crs = read_tif(file)
        return mask.astype(np.float32)

    def get_label(self, site: str, x: int, y: int) -> np.ndarray:
        label_name = self.cfg.DATASET.LABEL
        file = self.root_path / site / label_name / f'{site}_{label_name}_{y:010d}-{x:010d}.tif'
        img, geotransform, crs = read_tif(file)

        # already preprocessed label
        if label_name == 'burnseverity':
            return img.astype(np.float32)

        # thresholding the product based on config
        thresholds = self.cfg.DATASET.THRESHOLDS
        label = np.zeros(img.shape, dtype=np.float32)

        lower_bound = 10e-6
        for i, thresh in enumerate(thresholds):
            label[np.logical_and(lower_bound <= img, img < thresh)] = i
            lower_bound = thresh
        label[img >= thresholds[-1]] = len(thresholds)
        if self.cfg.DATASET.USE_FIREMASK:
            mask = self.get_firemask(site, x, y)
            label[np.logical_not(mask)] = 0
        return label

    def get_s2_feature_selection(self):
        available_features = self.cfg.DATASET.AVAILABLE_S2_BANDS
        selected_features = self.cfg.DATASET.S2_BANDS
        feature_selection = self.get_feature_selection(available_features, selected_features)
        return feature_selection

    def get_s1_feature_selection(self):
        available_bands = self.cfg.DATASET.AVAILABLE_S1_BANDS
        selected_bands = self.cfg.DATASET.S1_BANDS
        band_selection = self.get_feature_selection(available_bands, selected_bands)
        return band_selection

    @staticmethod
    def get_feature_selection(bands, selection):
        band_selection = [False for _ in range(len(bands))]
        for selected_band in selection:
            i = bands.index(selected_band)
            band_selection[i] = True
        return band_selection

    def get_n_features(self) -> int:
        mode = self.cfg.DATASET.MODE
        sar_features = len(self.s1_feature_selection)
        optical_features = len(self.s2_feature_selection)
        if self.cfg.DATASET.INCLUDE_PREFIRE_S2:
            optical_features *= 2

        if mode == 'sar':
            return sar_features
        if mode == 'optical':
            return optical_features
        else:
            return sar_features + optical_features


class TrainingDataset(AbstractDataset):
    def __init__(self, cfg, run_type: str, no_augmentation: bool = False):
        super().__init__(cfg)

        # TODO: change this to support multiple sites
        self.site = 'elephanthill'
        self.label_name = cfg.DATASET.LABEL

        # loading samples
        samples_file = self.root_path / self.site / f'{run_type}_samples.json'
        self.samples = load_json(samples_file)
        if cfg.DATASET.LABEL == 'dNBR' or cfg.DATASET.LABEL == 'rbr':
            self.samples = [s for s in self.samples if not s['has_masked_pixels']]
        self.length = len(self.samples)

        if no_augmentation:
            self.transform = transforms.Compose([Numpy2Torch()])
        else:
            self.transform = compose_transformations(cfg)

    def __getitem__(self, index: int) -> dict:

        sample = self.samples[index]
        site = sample['site']
        x = sample['x']
        y = sample['y']

        img = self.get_img(site, x, y)
        label = self.get_label(site, x, y)

        img, label = self.transform((img, label))

        item = {
            'img': img,
            'label': label,
            'site': site,
            'x': x,
            'y': y,
        }

        return item

    def __len__(self):
        return self.length

    def sampler(self):
        # TODO: investigate baseline
        weights = np.array([float(sample['weight']) for sample in self.samples])
        sampler = torch_data.WeightedRandomSampler(weights=weights, num_samples=self.length, replacement=True)
        return sampler


# dataset for classifying a scene
class InferenceDataset(AbstractDataset):

    def __init__(self, cfg, site: str, no_label: bool = False):
        super().__init__(cfg)

        self.site = site
        self.no_label = no_label
        self.transform = transforms.Compose([Numpy2Torch()])

        # getting all tiles
        tiles_file = self.root_path / site / 'samples.json'
        self.tiles = load_json(tiles_file)
        self.length = len(self.tiles)

        self.n_features = self.get_n_features()
        self.n_out = cfg.MODEL.OUT_CHANNELS

        tile, self.geotransform, self.crs = self.get_s2_data(site, 0, 0)
        self.tile_size, _, _ = tile.shape

        # getting patch ids and computing extent
        self.max_x = max([tile['x'] for tile in self.tiles])
        self.max_y = max([tile['y'] for tile in self.tiles])

    def __getitem__(self, index_center: int) -> dict:

        tile_center = self.tiles[index_center]
        x_center, y_center = tile_center['x'], tile_center['y']
        tile_size = self.tile_size

        extended_tile = np.zeros((3 * tile_size, 3 * tile_size, self.n_features), dtype=np.float32)
        extended_label = np.zeros((3 * tile_size, 3 * tile_size, 1), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                x = x_center + (j - 1) * tile_size
                y = y_center + (i - 1) * tile_size
                if self._tile_exists(x, y):
                    tile = self.get_img(self.site, x, y)

                    if self.no_label:
                        label = np.zeros((tile.shape[0], tile.shape[1], 1), dtype=np.float32)
                    else:
                        label = self.get_label(self.site, x, y)
                else:
                    tile = np.zeros((tile_size, tile_size, self.n_features), dtype=np.float32)
                    label = np.zeros((tile_size, tile_size, 1), dtype=np.float32)

                m, n, _ = tile.shape
                y_start = i * tile_size
                y_end = y_start + m
                x_start = j * tile_size
                x_end = x_start + n
                extended_tile[y_start:y_end, x_start:x_end, :] = tile
                extended_label[y_start:y_end, x_start:x_end, :] = label

        # dummy_label = np.zeros((extended_tile.shape[0], extended_tile.shape[1], 1), dtype=np.float32)
        extended_tile, extended_label = self.transform((extended_tile, extended_label))

        item = {
            'img': extended_tile,
            'label': extended_label,
            'x': x_center,
            'y': y_center,
        }

        return item

    def _tile_exists(self, x: int, y: int) -> bool:
        if 0 <= x <= self.max_x and 0 <= y <= self.max_y:
            return True
        return False

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.tile_size
        width = self.max_x + self.tile_size
        return np.zeros((height, width), dtype=dtype)

    def get_geo(self):
        return self.geotransform, self.crs

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} tiles.'
