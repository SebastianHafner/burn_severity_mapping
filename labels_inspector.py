import torch
import torchvision.transforms.functional as TF
from utils.geofiles import *
from utils.network import load_net
from experiment_manager.config import config
import numpy as np
from utils.datasets import TrainingDataset
import matplotlib.pyplot as plt
from utils.visualization import *
from tqdm import tqdm

ROOT_PATH = Path('/storage/shafner/burn_severity_mapping')
CONFIG_PATH = Path('/home/shafner/burn_severity_mapping/configs')


def show_data(config_name: str):

    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = TrainingDataset(cfg, 'train', no_augmentation=True)

    for index in tqdm(range(len(dataset))):
        fig, axs = plt.subplots(1, 2)
        sample = dataset.__getitem__(index)
        img = sample['img']
        label = sample['label']
        img = img.squeeze().numpy().transpose((1, 2, 0))
        label = label.squeeze().numpy()
        plot_burn_severity(cfg, axs[0], label)
        plot_s2(cfg, axs[1], img)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    config_name = 'baseline'
    show_data(config_name)

