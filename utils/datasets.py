import torch
from torch.utils import data as torch_data
from torchvision import transforms
import numpy as np
from utils.augmentations import *
from utils.geofiles import *
import cv2


class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, run_type: str, no_augmentation: bool = False):
        super().__init__()

        self.cfg = cfg
        self.root_path = Path(cfg.DATASET.PATH)
        self.site = 'elephanthill'

        # loading samples
        samples_file = self.root_path / self.site / f'{run_type}_samples.json'
        self.samples = load_json(samples_file)
        self.length = len(self.samples)

        if no_augmentation:
            self.transform = transforms.Compose([Numpy2Torch()])
        else:
            self.transform = compose_transformations(cfg)

        # feature selection
        self.s1_feature_selection = self.get_s1_feature_selection()
        self.s2_feature_selection = self.get_s2_feature_selection()

    def __getitem__(self, index) -> dict:

        sample = self.samples[index]
        i = sample['i']
        j = sample['j']

        mode = self.cfg.DATASET.MODE
        if mode == 'optical':
            img = self.get_s2_data(i, j)
        elif mode == 'sar':
            img = self.get_s1_data(i, j)
        else:
            s1_img = self.get_s1_data(i, j)
            s2_img = self.get_s2_data(i, j)
            img = np.concatenate((s1_img, s2_img), axis=-1)

        label = self.get_label(i, j)

        img, label = self.transform((img, label))

        item = {
            'img': img,
            'label': label,
            'site': sample['site'],
            'i': i,
            'j': j,
        }

        return item

    def get_s2_data(self, i: int, j: int) -> np.ndarray:
        lvl = self.cfg.DATASET.S2_PROCESSING_LEVEL
        file = self.root_path / self.site / f's2{lvl}' / f'{self.site}_s2{lvl}{i:010d}-{j:010d}.tif'
        img, _, _ = read_tif(file)
        img = img[:, :, self.s2_feature_selection]
        return img.astype(np.float32)

    def get_s1_data(self, i: int, j: int) -> np.ndarray:
        file = self.root_path / self.site / f's1' / f'{self.site}_s1{i:010d}-{j:010d}.tif'
        img, _, _ = read_tif(file)
        img = img[:, :, self.s1_feature_selection]
        return img.astype(np.float32)

    def get_label(self, i: int, j: int) -> np.ndarray:
        label = self.cfg.DATASET.LABEL
        file = self.root_path / self.site / label / f'{self.site}_{label}{i:010d}-{j:010d}.tif'
        img, _, _ = read_tif(file)

        thresholds = self.cfg.DATASET.THRESHOLDS
        for i, thresh in enumerate(thresholds):
            img[img < thresh] = i
        img[img >= thresholds[-1]] = len(thresholds)

        return img.astype(np.float32)

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

    @ staticmethod
    def get_feature_selection(bands, selection):
        band_selection = [False for _ in range(len(bands))]
        for selected_band in selection:
            i = bands.index(selected_band)
            band_selection[i] = True
        return band_selection

    def __len__(self):
        return self.length

    def sampler(self):
        # TODO: investigate baseline
        weights = np.array([float(sample['weight']) for sample in self.samples])
        sampler = torch_data.WeightedRandomSampler(weights=weights, num_samples=self.length, replacement=True)
        return sampler


# dataset for classifying a scene
class TilesInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, year: int):
        super().__init__()

        self.cfg = cfg
        self.year = year
        self.root_path = Path(cfg.DATASET.PATH) / 'time_series' / str(year)
        self.transform = transforms.Compose([Numpy2Torch()])

        # getting all files
        files = [f for f in self.root_path.glob('**/*')]
        self.length = len(files)

        available_bands = cfg.DATASET.SATELLITE.AVAILABLE_S2_BANDS
        selected_bands = cfg.DATASET.SATELLITE.S2_BANDS
        self.feature_selection = self._get_feature_selection(available_bands, selected_bands)
        self.n_features = len(selected_bands)

        self.n_out = len(cfg.DATASET.LABEL.REMAPPER.REMAPPED_CLASSES)

        self.basename = basename_from_file(files[0])
        patch_id = f'{0:010d}-{0:010d}'
        patch, self.geotransform, self.crs = self._load_img(patch_id)
        self.patch_size, _, _ = patch.shape

        # getting patch ids and computing extent
        self.patch_ids = [file2id(f) for f in files]
        self.coords = [id2yx(patch_id) for patch_id in self.patch_ids]
        self.max_y = max([c[0] for c in self.coords])
        self.max_x = max([c[1] for c in self.coords])

    def __getitem__(self, patch_id_center):

        y_center, x_center = id2yx(patch_id_center)

        extended_patch = np.zeros((3 * self.patch_size, 3 * self.patch_size, self.n_features), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                y = y_center + (i - 1) * self.patch_size
                x = x_center + (j - 1) * self.patch_size
                patch_id = f'{y:010d}-{x:010d}'
                if self._patch_id_exists(patch_id):
                    patch, _, _ = self._load_img(patch_id)
                else:
                    patch = np.zeros((self.patch_size, self.patch_size, self.n_features), dtype=np.float32)
                m, n, _ = patch.shape
                i_start = i * self.patch_size
                i_end = i_start + m
                j_start = j * self.patch_size
                j_end = j_start + n
                extended_patch[i_start:i_end, j_start:j_end, :] = patch

        dummy_label = np.zeros((extended_patch.shape[0], extended_patch.shape[1], 1), dtype=np.float32)
        extended_patch, _ = self.transform((extended_patch, dummy_label))

        item = {
            'x': extended_patch,
            'i': y_center,
            'j': x_center,
            'patch_id': patch_id_center,
        }

        return item

    def _load_img(self, patch_id):
        file = self.root_path / f'{self.basename}_{patch_id}.tif'
        img, transform, crs = read_tif(file)
        img = img[:, :, self.feature_selection] / self.cfg.DATASET.SATELLITE.S2_RESCALE_FACTOR
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _patch_id_exists(self, patch_id):
        return True if patch_id in self.patch_ids else False

    @ staticmethod
    def _get_feature_selection(bands, selection):
        band_selection = [False for _ in range(len(bands))]
        for selected_band in selection:
            i = bands.index(selected_band)
            band_selection[i] = True
        return band_selection

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.patch_size
        width = self.max_x + self.patch_size
        return np.zeros((height, width, self.n_out), dtype=dtype)

    def get_geo(self):
        return self.geotransform, self.crs

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} tiles.'


# dataset for classifying a scene
class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, s2_file: Path = None, patch_size: int = 256):
        super().__init__()

        self.cfg = cfg
        self.s2_file = s2_file

        arr, self.geotransform, self.crs = read_tif(s2_file)
        self.height, self.width, _ = arr.shape

        self.patch_size = patch_size
        self.rf = 8
        self.n_rows = (self.height - self.rf) // patch_size
        self.n_cols = (self.width - self.rf) // patch_size
        self.length = self.n_rows * self.n_cols

        self.transform = transforms.Compose([Numpy2Torch()])

        s2_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        selected_bands = cfg.DATASET.SATELLITE.SENTINEL2_BANDS
        self.band_selection = self.get_band_selection(s2_bands, selected_bands)



    def __getitem__(self, index):

        i_start = index // self.n_cols * self.patch_size
        j_start = index % self.n_cols * self.patch_size
        # check for border cases and add padding accordingly
        # top left corner
        if i_start == 0 and j_start == 0:
            i_end = self.patch_size + 2 * self.rf
            j_end = self.patch_size + 2 * self.rf
        # top
        elif i_start == 0:
            i_end = self.patch_size + 2 * self.rf
            j_end = j_start + self.patch_size + self.rf
            j_start -= self.rf
        elif j_start == 0:
            j_end = self.patch_size + 2 * self.rf
            i_end = i_start + self.patch_size + self.rf
            i_start -= self.rf
        else:
            i_end = i_start + self.patch_size + self.rf
            i_start -= self.rf
            j_end = j_start + self.patch_size + self.rf
            j_start -= self.rf

        img = self._get_sentinel_data(i_start, i_end, j_start, j_end)
        img, _ = self.transform((img, np.empty((1, 1, 1))))
        patch = {
            'x': img,
            'row': (i_start, i_end),
            'col': (j_start, j_end)
        }

        return patch

    def _get_sentinel_data(self, i_start: int, i_end: int, j_start: int, j_end: int):
        img_patch = self.img[i_start:i_end, j_start:j_end, ]
        return np.nan_to_num(img_patch).astype(np.float32)

    def _get_feature_selection(self, features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def __len__(self):
        return self.length



