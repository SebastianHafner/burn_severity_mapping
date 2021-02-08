from utils.geofiles import *
import numpy as np
from affine import Affine

# TODO: include weights for classes in sample files

ROOT_PATH = Path('/storage/shafner/slum_extent_mapping/kigali_dataset')


def is_square(file: Path) -> bool:
    arr, _, _ = read_tif(file)
    height, width, _ = arr.shape
    return True if height == width else False


def create_samples_files(path: Path, label_name: str, split: float = 0.1, seed: int = 7):

    label_folder = path / label_name
    label_files = [f for f in label_folder.glob('**/*')]

    n_samples = len(label_files)

    np.random.seed(seed)
    random_numbers = np.random.rand(n_samples)

    samples_train = []
    samples_validation = []
    for file, rand in zip(label_files, random_numbers):
        if not is_square(file):
            continue
        sensor, roi, coords = file.stem.split('_')
        i, j = coords.split('-')

        sample = {
            'roi': roi,
            'i': int(i),
            'j': int(j),
        }

        if rand > split:
            samples_train.append(sample)
        else:
            samples_validation.append(sample)

    write_json(ROOT_PATH / f'train_samples.json', samples_train)
    write_json(ROOT_PATH / f'validation_samples.json', samples_validation)


# only works for southern hemisphere
def create_tiles(file: Path, tile_size: int = 64):
    output_path = file.parent
    base_name = file.stem

    arr, transform, crs = read_tif(file)
    height, width, _ = arr.shape
    n_rows, n_cols = height // tile_size, width // tile_size
    img_res_x, _, img_min_x, _, img_res_y, img_max_y, *_ = transform

    # earth engine output is row col
    for i in range(n_rows):
        for j in range(n_cols):
            i_start, i_end = i * tile_size, (i + 1) * tile_size
            j_start, j_end = j * tile_size, (j + 1) * tile_size
            tile_arr = arr[i_start:i_end, j_start:j_end, ]

            tile_min_x = img_min_x + j_start * img_res_x
            tile_max_y = img_max_y + i_start * img_res_y
            tile_transform = Affine(img_res_x, 0, tile_min_x, 0, img_res_y, tile_max_y)

            file = output_path / f'{base_name}_{i_start:010d}-{j_start:010d}.tif'
            write_tif(file, tile_arr, tile_transform, crs)


if __name__ == '__main__':

    # create_tiles(ROOT_PATH / 'sentinel2' / 'sentinel2_aoi1.tif')
    # create_tiles(ROOT_PATH / 'sentinel2' / 'sentinel2_aoi2.tif')
    # create_tiles(ROOT_PATH / 'classfractions' / 'classfractions_aoi1.tif')
    # create_tiles(ROOT_PATH / 'classfractions' / 'classfractions_aoi2.tif')
    create_samples_files(ROOT_PATH, 'landcover')
