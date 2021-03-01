from utils.geofiles import *
import numpy as np
from tqdm import tqdm


ROOT_PATH = Path('/storage/shafner/burn_severity_mapping/wildfire_dataset')


def is_square(file: Path) -> bool:
    arr, _, _ = read_tif(file)
    height, width, _ = arr.shape
    return True if height == width else False


def has_masked_pixels(file: Path) -> bool:
    arr, _, _ = read_tif(file)
    if np.sum(arr):
        return False
    return True


def create_samples_files(path: Path, site: str, split: float = 0.1, seed: int = 7):
    print(f'preprocessing {site}...')

    mask_folder = path / site / 'mask'
    mask_files = [f for f in mask_folder.glob('**/*')]

    np.random.seed(seed)
    random_numbers = np.random.rand(len(mask_files))

    samples_train, samples_validation = [], []
    for file, rand in tqdm(zip(mask_files, random_numbers)):
        coords = file.stem.split('_')[-1]
        y, x = coords.split('-')
        sample = {
            'site': site,
            'x': int(x),
            'y': int(y),
            'has_masked_pixels': has_masked_pixels(file)
        }

        if is_square(file):
            if rand > split:
                samples_train.append(sample)
            else:
                samples_validation.append(sample)

    write_json(ROOT_PATH / site / f'train_samples.json', samples_train)
    write_json(ROOT_PATH / site / f'validation_samples.json', samples_validation)


def create_inference_files(path: Path, site: str, input_data: str = 's2toa'):
    print(f'preprocessing {site}...')

    folder = path / site / input_data
    files = [f for f in folder.glob('**/*')]

    samples = []
    for file in tqdm(files):
        coords = file.stem.split('_')[-1]
        y, x = coords.split('-')
        sample = {
            'site': site,
            'x': int(x),
            'y': int(y),
            'has_masked_pixels': has_masked_pixels(file)
        }

        samples.append(sample)
    write_json(ROOT_PATH / site / f'tiles.json', samples)


if __name__ == '__main__':

    sites = ['axingmyrkullen', 'brattsjo', 'elephanthill', 'fagelsjo']
    sites = ['fagelsjo', 'elephanthill', 'trangslet']
    for site in sites:
        create_samples_files(ROOT_PATH, site)
