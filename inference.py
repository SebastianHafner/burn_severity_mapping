import torch
import torchvision.transforms.functional as TF
from utils.geofiles import *
from utils.network import load_net
from experiment_manager.config import config
import numpy as np
from utils.datasets import InferenceDataset
import matplotlib.pyplot as plt
from utils.visualization import *
from tqdm import tqdm


ROOT_PATH = Path('/storage/shafner/burn_severity_mapping')
CONFIG_PATH = Path('/home/shafner/burn_severity_mapping/configs')
NETWORK_PATH = Path('/storage/shafner/burn_severity_mapping/networks')
INFERENCE_PATH = Path('/storage/shafner/burn_severity_mapping/inference')


def site_inference(config_name: str, site: str):

    # loading cfg and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net = load_net(cfg, NETWORK_PATH / f'{config_name}_{cfg.CHECKPOINT}.pkl')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    dataset = InferenceDataset(cfg, site, no_label=True)

    # config inference directory
    save_path = ROOT_PATH / 'inference' / config_name
    save_path.mkdir(exist_ok=True)

    prediction = dataset.get_arr()
    transform, crs = dataset.get_geo()

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            tile = dataset.__getitem__(index)
            img = tile['img'].to(device)
            logits = net(img.unsqueeze(0))
            sm = torch.nn.Softmax(dim=1)
            prob = sm(logits)
            pred = torch.argmax(prob, dim=1)

            pred = pred.squeeze().cpu().numpy().astype('uint8')

            center_pred = pred[dataset.tile_size:dataset.tile_size*2, dataset.tile_size:dataset.tile_size*2, ]

            y_start = tile['y']
            y_end = y_start + dataset.tile_size
            x_start = tile['x']
            x_end = x_start + dataset.tile_size
            prediction[y_start:y_end, x_start:x_end] = center_pred

    output_file = save_path / f'pred_{site}_{config_name}.tif'
    write_tif(output_file, prediction[:, :, None], transform, crs)


def site_label(config_name: str, site: str):

    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = InferenceDataset(cfg, site)

    # config inference directory
    save_path = ROOT_PATH / 'inference' / config_name
    save_path.mkdir(exist_ok=True)

    label = dataset.get_arr()
    transform, crs = dataset.get_geo()

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            tile = dataset.__getitem__(index)
            extended_label = tile['label']
            extended_label = extended_label.squeeze().numpy().astype('uint8')
            center_label = extended_label[dataset.tile_size:dataset.tile_size * 2, dataset.tile_size:dataset.tile_size * 2]

            y_start = tile['y']
            y_end = y_start + dataset.tile_size
            x_start = tile['x']
            x_end = x_start + dataset.tile_size
            label[y_start:y_end, x_start:x_end] = center_label

    output_file = save_path / f'label_{site}_{config_name}.tif'
    write_tif(output_file, label[:, :, None], transform, crs)


if __name__ == '__main__':
    config_name = 'rbr'
    sites = ['elephanthill', 'axingmyrkullen', 'brattsjo', 'fagelsjo']
    site_inference(config_name, 'fagelsjo')
    # for site in sites:
    #     site_inference(config_name, site)
    # site_inference('baseline_sar', 'axingmyrkullen')
    # site_label(config_name, 'elephanthill')
