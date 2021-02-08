import torch
import torchvision.transforms.functional as TF
from utils.geofiles import *
from utils.network import load_net
from experiment_manager.config import config
import numpy as np
from utils.datasets import LandCoverDataset, TilesInferenceDataset
import matplotlib.pyplot as plt
from utils.visualization import *
from tqdm import tqdm


ROOT_PATH = Path('/storage/shafner/slum_extent_mapping')
CONFIG_PATH = Path('/home/shafner/slum_extent_mapping/configs')
NETWORK_PATH = Path('/storage/shafner/slum_extent_mapping/networks')
INFERENCE_PATH = Path('/storage/shafner/slum_extent_mapping/inference')


def end_to_end_inference(config_name: str, checkpoint: int, s2_file: Path, save_output: bool = False):

    # loading cfg and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net = load_net(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    # loading network and sentinel-2 img
    raw_img, transform, crs = read_tif(s2_file)
    feature_selection = LandCoverDataset.get_feature_selection(cfg.DATASET.SATELLITE.AVAILABLE_S2_BANDS,
                                                               cfg.DATASET.SATELLITE.S2_BANDS)
    img = raw_img[:, :, feature_selection] / cfg.DATASET.SATELLITE.S2_RESCALE_FACTOR
    img_tensor = TF.to_tensor(img).float().to(device)

    # classifying image
    logits = net(img_tensor.unsqueeze(0))
    sm = torch.nn.Softmax(dim=1)
    prob = sm(logits)
    pred = torch.argmax(prob, dim=1)

    prob = prob.squeeze().float().detach().cpu().numpy()
    prob = prob.transpose((1, 2, 0))
    pred = pred.squeeze().float().detach().cpu().numpy()

    if save_output:
        save_path = INFERENCE_PATH / config_name
        save_path.mkdir(exist_ok=True)
        # prediction argmax of prob
        pred_file = save_path / f'pred_{config_name}_{s2_file.stem}.tif'
        write_tif(pred_file, np.expand_dims(pred, axis=-1), transform, crs)
        # prob
        prob_file = save_path / f'prob_{config_name}_{s2_file.stem}.tif'
        write_tif(prob_file, prob, transform, crs)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        plot_s2(cfg, axs[0], raw_img, title=f'{s2_file.stem}')
        plot_classification(cfg, axs[1], pred, title=f'{config_name}_{s2_file.stem}')
        plt.show()


def show_prediction(config_name: str, file: Path):
    # loading cfg and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    prediction, _, _ = read_tif(file)
    plot_classification(cfg, ax, prediction, title=f'{config_name}')
    plt.show()
    plt.close()


# t1 and t2 file are class probability outputs
def compute_change(t1_file: Path, t2_file: Path, class_index: int, output_file: Path = None):
    t1_prob, transform, crs = read_tif(t1_file)
    t2_prob, _, _ = read_tif(t2_file)
    t1_class_prob = t1_prob[:, :, class_index]
    t2_class_prob = t2_prob[:, :, class_index]
    diff = np.abs(t1_class_prob - t2_class_prob)
    write_tif(output_file, np.expand_dims(diff, axis=-1), transform, crs)


def end_to_end_inference_new(config_name: str, checkpoint: int, year: int):

    # loading cfg and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net = load_net(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    dataset = TilesInferenceDataset(cfg, year)

    # config inference directory
    save_path = ROOT_PATH / 'inference' / config_name
    save_path.mkdir(exist_ok=True)

    prob_output = dataset.get_arr()
    transform, crs = dataset.get_geo()

    with torch.no_grad():
        for patch_id in tqdm(dataset.patch_ids):
            patch = dataset.__getitem__(patch_id)
            img = patch['x'].to(device)
            logits = net(img.unsqueeze(0))
            sm = torch.nn.Softmax(dim=1)
            prob = sm(logits) * 100

            prob = prob.squeeze().detach().cpu().numpy().astype('uint8')
            prob = np.clip(prob, 0, 100)
            prob = prob.transpose((1, 2, 0))

            center_prob = prob[dataset.patch_size:dataset.patch_size*2, dataset.patch_size:dataset.patch_size*2, ]

            i_start = patch['i']
            i_end = i_start + dataset.patch_size
            j_start = patch['j']
            j_end = j_start + dataset.patch_size
            prob_output[i_start:i_end, j_start:j_end, :] = center_prob

    output_file = save_path / f'prob_{dataset.basename}_{config_name}.tif'
    write_tif(output_file, prob_output, transform, crs)


if __name__ == '__main__':

    config_name = 'builtup_hard'
    checkpoint = 100
    for year in [2016, 2017, 2018, 2019, 2020]:
        end_to_end_inference_new(config_name, checkpoint, year)
    # time series inference
    # for year in [2016, 2020]:
    #     folder = ROOT_PATH / 'kigali_dataset' / 'time_series' / str(year)
    #     s2_tiles = [f for f in folder.glob('**/*')]
    #     for s2_tile in s2_tiles:
    #         end_to_end_inference(config_name, checkpoint, s2_file=s2_tile, save_output=True)
    #
    #     for output in ['prob', 'pred']:
    #         tiles_path = INFERENCE_PATH / config_name
    #         file_name = f'{output}_{config_name}_sentinel2_{year}'
    #         combine_tiff_tiles(tiles_path, file_name, delete_tiles=True)

    # t1_file = INFERENCE_PATH / config_name / f'mosaic_prob_{config_name}_sentinel2_2016.tif'
    # t2_file = INFERENCE_PATH / config_name / f'mosaic_prob_{config_name}_sentinel2_2020.tif'
    # change_file = ROOT_PATH / 'change' / f'urban_change_{config_name}.tif'
    # compute_change(t1_file, t2_file, 0, change_file)

    # for year in [2016, 2020]:
    #     folder = ROOT_PATH / 'kigali_dataset' / 'time_series' / str(year)
    #     file_name = f'sentinel2_{year}'
    #     combine_tiff_tiles(folder, file_name, delete_tiles=False)

