from utils.geofiles import *
import numpy as np


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


def create_samples_files(path: Path, site: str, split: float = 0.3, seed: int = 7):

    s2_folder = path / site / 's2toa'
    s2_files = [f for f in s2_folder.glob('**/*')]

    np.random.seed(seed)
    random_numbers = np.random.rand(len(s2_files))

    samples_train, samples_validation, samples = [], [], []
    for file, rand in zip(s2_files, random_numbers):
        coords = file.stem[-21:]
        y, x = coords.split('-')
        sample = {
            'site': site,
            'x': int(x),
            'y': int(y),
        }

        if is_square(file):
            if rand > split:
                samples_train.append(sample)
            else:
                samples_validation.append(sample)
        samples.append(sample)

    write_json(ROOT_PATH / site / f'train_samples.json', samples_train)
    write_json(ROOT_PATH / site / f'validation_samples.json', samples_validation)
    write_json(ROOT_PATH / site / f'samples.json', samples)


if __name__ == '__main__':

    sites = ['axingmyrkullen', 'brattsjo', 'elephanthill', 'fagelsjo']
    for site in sites:
        create_samples_files(ROOT_PATH, site)
