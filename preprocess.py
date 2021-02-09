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


def create_samples_files(path: Path, site: str, split: float = 0.2, seed: int = 7):

    mask_folder = path / site / 'mask'
    mask_files = [f for f in mask_folder.glob('**/*')]

    np.random.seed(seed)
    random_numbers = np.random.rand(len(mask_files))

    samples_train = []
    samples_validation = []
    for file, rand in zip(mask_files, random_numbers):
        if not is_square(file) or has_masked_pixels(file):
            continue
        coords = file.stem[-21:]
        i, j = coords.split('-')

        sample = {
            'site': site,
            'i': int(i),
            'j': int(j),
        }

        if rand > split:
            samples_train.append(sample)
        else:
            samples_validation.append(sample)

    write_json(ROOT_PATH / site / f'train_samples.json', samples_train)
    write_json(ROOT_PATH / site / f'validation_samples.json', samples_validation)


if __name__ == '__main__':

    create_samples_files(ROOT_PATH, 'elephanthill')
