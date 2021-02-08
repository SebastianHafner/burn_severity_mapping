import torch
from torch.utils import data as torch_data
from torchvision import transforms
import numpy as np
from utils.augmentations import *
from utils.geofiles import *
import cv2


class LandCoverDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset: str, no_augmentation: bool = False):
        super().__init__()

        self.cfg = cfg
        self.root_path = Path(cfg.DATASET.PATH)
        self.label_name = cfg.DATASET.LABEL.NAME

        # loading samples
        samples_file = self.root_path / f'{dataset}_samples.json'
        self.samples = load_json(samples_file)
        self.length = len(self.samples)

        if no_augmentation:
            self.transform = transforms.Compose([Numpy2Torch()])
        else:
            self.transform = compose_transformations(cfg)

        # feature selection
        self.satellite = cfg.DATASET.SATELLITE
        self.satellite_name = cfg.DATASET.SATELLITE.NAME
        self.s2_feature_selection = self.get_s2_feature_selection()
        self.wv_feature_selection = self.get_wv_feature_selection()

        # handling slum inclusion and remapping
        self.original_classes = list(cfg.DATASET.LABEL.CLASSES)
        self.include_slums = cfg.DATASET.LABEL.INCLUDE_SLUMS
        if self.include_slums:
            self.slum_classes = cfg.DATASET.LABEL.SLUM_CLASSES
            self.slum_index = len(self.original_classes)
            self.original_classes.append('slums')
        self.remapper = cfg.DATASET.LABEL.REMAPPER
        self.n_classes = len(self.remapper.REMAPPED_CLASSES)

        self.res_factor = cfg.DATASET.SATELLITE.S2_RES / cfg.DATASET.SATELLITE.WW_RES

    def __getitem__(self, index, enforce_hard_labels: bool = False):

        sample = self.samples[index]
        roi = sample['roi']
        i = sample['i']
        j = sample['j']

        if self.satellite_name == 'sentinel2':
            img = self.get_s2_data(roi, i, j)
        else:
            img = self.get_wv_data(roi, i, j)

        if self.label_name == 'classfractions':
            label = self.get_classfractions_data(roi, i, j)
            if not self.cfg.DATASET.LABEL.SOFT_LABELS:  # one-hot encoding
                label_hard = np.argmax(label, axis=-1)
                label = (np.arange(self.n_classes) == label_hard[..., None]).astype(np.float32)
        else:
            label = self.get_landcover_data(roi, i, j)

        img, label = self.transform((img, label))

        item = {
            'img': img,
            'label': label,
            'roi': roi,
            'i': i,
            'j': j,
        }

        return item

    def get_raw_s2_data(self, roi: str, i: int, j: int) -> np.ndarray:
        i_rescaled, j_rescaled = int(i // self.res_factor), int(j // self.res_factor)
        file = self.root_path / 'sentinel2' / f'sentinel2_{roi}_{i_rescaled:010d}-{j_rescaled:010d}.tif'
        img, _, _ = read_tif(file)
        return img.astype(np.float32)

    def get_s2_data(self, roi: str, i: int, j: int) -> np.ndarray:
        img = self.get_raw_s2_data(roi, i, j)
        return img[:, :, self.s2_feature_selection] / self.satellite.S2_RESCALE_FACTOR

    def get_raw_wv_data(self, roi: str, i: int, j: int) -> np.ndarray:
        file = self.root_path / 'worldview' / f'worldview_{roi}_{i:010d}-{j:010d}.tif'
        img, _, _ = read_tif(file)
        return img.astype(np.float32)

    def get_wv_data(self, roi: str, i: int, j: int) -> np.ndarray:
        img = self.get_raw_wv_data(roi, i, j)
        return img[:, :, self.wv_feature_selection] / self.satellite.WW_RESCALE_FACTOR

    def get_raw_landcover_data(self, roi: str, i: int, j: int) -> np.ndarray:
        landcover_file = self.root_path / 'landcover' / f'landcover_{roi}_{i:010d}-{j:010d}.tif'
        landcover, _, _ = read_tif(landcover_file)
        return landcover

    def get_landcover_data(self, roi: str, i: int, j: int) -> np.ndarray:
        # TODO: handle resolution mismatch
        landcover = self.get_raw_landcover_data(roi, i, j)
        # if slums are included, all specified classes (slum classes) that are withing slums are remapped to slums
        # slums are then added as a class to the original ones
        if self.include_slums:
            # TODO: this is broken because slums exist only for roi1
            slum_extent = self.get_slum_extent(i, j)
            for i, slum_class in enumerate(self.slum_classes):
                class_index = self.original_classes.index(slum_class)
                class_withing_slums = np.logical_and(landcover == class_index, slum_extent)
                landcover[class_withing_slums] = self.slum_index
        landcover = self.remap_classes(landcover)
        return landcover.astype(np.float32)

    def get_raw_classfractions_data(self, roi: str, i: int, j: int) -> np.ndarray:
        i_rescaled, j_rescaled = int(i // self.res_factor), int(j // self.res_factor)
        file = self.root_path / 'classfractions' / f'classfractions_{roi}_{i_rescaled:010d}-{j_rescaled:010d}.tif'
        fractions, _, _ = read_tif(file)
        return fractions.astype(np.float32)

    def get_classfractions_data(self, roi: str, i: int, j: int) -> np.ndarray:
        fractions = self.get_raw_classfractions_data(roi, i, j)
        # if slums are included, we compute the fraction of slums for each pixel. The fraction is computed based
        # on the class fractions of classes comprising slums
        if self.include_slums:
            # TODO: this is broken because slums only exist for roi1
            # loading slum extent
            slum_extent = self.get_slum_extent(i, j, downsample=True)

            # new band for slum fraction
            m, n, _ = fractions.shape
            slum_fraction = np.zeros((m, n))

            for i, class_ in enumerate(self.slum_classes):
                class_index = self.original_classes.index(class_)
                class_fraction = fractions[:, :, class_index]
                slum_fraction = slum_fraction + class_fraction * slum_extent
                class_fraction[slum_extent] = 0

            fractions = np.concatenate([fractions, np.expand_dims(slum_fraction, axis=-1)], axis=-1)

        fractions_remapped = self.remap_fractions(fractions)

        return fractions_remapped.astype(np.float32)

    # TODO: this is broken because slums only exist for roi1
    def get_slum_extent(self, i: int, j: int, downsample: bool = False) -> np.ndarray:
        file = self.root_path / 'slums' / f'slums_{i:010d}-{j:010d}.tif'
        slums, _, _ = read_tif(file)
        if downsample:
            m, n, _ = slums.shape
            to_m, to_n = int(m // self.res_factor), int(n // self.res_factor)
            slums = cv2.resize(slums, dsize=(to_m, to_n), interpolation=cv2.INTER_NEAREST)
        return slums.astype(np.bool)

    def remap_fractions(self, fractions: np.ndarray) -> np.ndarray:
        new_classes = self.remapper.REMAPPED_CLASSES
        height, width, _ = fractions.shape
        fractions_remapped = np.zeros((height, width, len(new_classes)))

        # map all origin classes to new classes according to the remap functions
        for i, class_name in enumerate(self.original_classes):
            new_class_name = self.remapper.REMAP_FUNC[i]
            j = new_classes.index(new_class_name)
            fractions_remapped[:, :, j] = fractions_remapped[:, :, j] + fractions[:, :, i]
        return fractions_remapped

    def remap_classes(self, label: np.ndarray) -> np.ndarray:
        label_remapped = np.empty(label.shape)
        new_classes = self.remapper.REMAPPED_CLASSES
        # map all origin classes to new classes according to the remap functions
        for i, class_name in enumerate(self.original_classes):
            new_class_name = self.remapper.REMAP_FUNC[i]
            j = new_classes.index(new_class_name)
            label_remapped[label == i] = j
        return label

    def get_s2_feature_selection(self):
        available_features = self.satellite.AVAILABLE_S2_BANDS
        selected_features = self.satellite.S2_BANDS
        feature_selection = self.get_feature_selection(available_features, selected_features)
        return feature_selection

    def get_wv_feature_selection(self):
        available_bands = self.satellite.AVAILABLE_WV_BANDS
        selected_bands = self.satellite.WV_BANDS
        band_selection = self.get_feature_selection(available_bands, selected_bands)
        return band_selection

    @ staticmethod
    def get_feature_selection(bands, selection):
        band_selection = [False for _ in range(len(bands))]
        for selected_band in selection:
            i = bands.index(selected_band)
            band_selection[i] = True
        return band_selection

    @staticmethod
    def feature_names(bands: list, metrics: list):
        # metric outer loop, bands inner loop
        return [f'{band}_{metric}' for band in bands for metric in metrics]

    def __len__(self):
        return self.length

    def compute_class_weights(self) -> list:
        # TODO: debug this
        n_class_pixels = np.zeros(self.n_classes)
        bins = np.arange(-0.5, self.n_classes, 1)
        for sample in self.samples:
            i = sample['i']
            j = sample['j']
            landcover = self.get_landcover_data(i, j)
            landcover_remapped = self.remap_classes(landcover)
            hist_sample, _ = np.histogram(landcover_remapped, bins=bins)
            n_class_pixels += hist_sample
        return n_class_pixels / np.sum(n_class_pixels)

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



