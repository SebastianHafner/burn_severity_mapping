import matplotlib.pyplot as plt
from experiment_manager.config import config
from utils.network import load_net
from utils.visualization import *
from utils.datasets import LandCoverDataset
import torch
from torch.utils import data as torch_data
from pathlib import Path
from tqdm import tqdm


ROOT_PATH = Path('/storage/shafner/slum_extent_mapping')
CONFIG_PATH = Path('/home/shafner/slum_extent_mapping/configs')
NETWORK_PATH = Path('/storage/shafner/slum_extent_mapping/networks')


def show_predictions(config_name: str, checkpoint: int, run_type: str, n: int, save_plots: bool = False):

    # loading cfg and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net = load_net(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    dataset = LandCoverDataset(cfg, run_type, no_augmentation=True)
    dataloader = torch_data.DataLoader(dataset, batch_size=1)

    for i, sample in enumerate(dataloader):
        img = sample['img'].to(device)
        i = int(sample['i'][0])
        j = int(sample['j'][0])
        roi = str(sample['roi'][0])

        fig, axs = plt.subplots(1, 5, figsize=(12, 4))

        # plotting sentinel-2 and worldview image
        img_wv = dataset.get_raw_wv_data(roi, i, j)
        plot_wv(cfg, axs[0], img_wv, title='worldview')
        img_s2 = dataset.get_raw_s2_data(roi, i, j)
        plot_s2(cfg, axs[3], img_s2, title='sentinel2')

        # classifying image
        with torch.no_grad():
            logits = net(img)
            sm = torch.nn.Softmax(dim=1)
            prob = sm(logits)
            pred = torch.argmax(prob, dim=1)
            pred = pred.float().detach().cpu().numpy().squeeze()

        plot_classification(cfg, axs[4], pred, 'prediction')

        landcover = dataset.get_raw_landcover_data(roi, i, j)
        plot_original_classes(cfg, axs[1], landcover, 'high res label')

        label = np.argmax(dataset.get_classfractions_data(roi, i, j), axis=-1)
        plot_classification(cfg, axs[2], label, 'label')

        if not save_plots:
            plt.show()
        else:
            output_path = ROOT_PATH / 'validation' / 'plots' / config_name
            output_path.mkdir(exist_ok=True)
            file = output_path / f'validation_{config_name}_{i:010d}_{j:010d}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        plt.close()

        if i + 1 == n:
            break


def show_probabilities(config_name: str, checkpoint: int, run_type: str, n: int, class_name: str,
                       save_plots: bool = False):
    # loading cfg and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net = load_net(cfg, NETWORK_PATH / f'{config_name}_{checkpoint}.pkl')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    dataset = LandCoverDataset(cfg, run_type, no_augmentation=True)
    dataloader = torch_data.DataLoader(dataset, batch_size=1)
<<<<<<< HEAD

    for i, sample in enumerate(dataloader):
=======
    class_index = cfg.DATASET.LABEL.REMAPPER.REMAPPED_CLASSES.index(class_name)

    for i, sample in enumerate(tqdm(dataloader)):
>>>>>>> a5841643530ed1859b3b894843649e2c1f36074e
        img = sample['img'].to(device)
        i = int(sample['i'][0])
        j = int(sample['j'][0])
        roi = str(sample['roi'][0])

        fig, axs = plt.subplots(1, 5, figsize=(12, 4))

<<<<<<< HEAD
        # plotting sentinel-2 and worldview image
        img_wv = dataset.get_raw_wv_data(roi, i, j)
        plot_wv(cfg, axs[0], img_wv, title='worldview')
        img_s2 = dataset.get_raw_s2_data(roi, i, j)
        plot_s2(cfg, axs[2], img_s2, title='sentinel2')
=======
        landcover = dataset.get_raw_landcover_data(roi, i, j)
        plot_original_classes(cfg, axs[0], landcover, 'high res label')

        label = dataset.get_classfractions_data(roi, i, j)
        label = label[:, :, class_index]
        plot_probabilities(axs[1], label, 'label')

        # plotting sentinel-2 and worldview image
        img_wv = dataset.get_raw_wv_data(roi, i, j)
        plot_wv(cfg, axs[2], img_wv, title='worldview')
        img_s2 = dataset.get_raw_s2_data(roi, i, j)
        plot_s2(cfg, axs[3], img_s2, title='sentinel2')


>>>>>>> a5841643530ed1859b3b894843649e2c1f36074e

        # classifying image
        with torch.no_grad():
            logits = net(img)
            sm = torch.nn.Softmax(dim=1)
            prob = sm(logits)
<<<<<<< HEAD
            prob = prob.detach().cpu().numpy().squeeze()
            class_index = cfg.DATASET.LABEL.REMAPPER.REMAPPED_CLASSES.index(class_name)
            class_prob = prob[class_index, ]

        plot_probability(axs[4], class_prob, f'{class_name} pred prob')

        landcover = dataset.get_raw_landcover_data(roi, i, j)
        plot_original_classes(cfg, axs[1], landcover, 'high res label')

        label = dataset.get_classfractions_data(roi, i, j)
        label = label[:, :, class_index]
        plot_probability(axs[3], label, f'{class_name} prob')
=======
            prob = prob.float().detach().cpu().numpy().squeeze()
            prob = prob[class_index, ]

        plot_probabilities(axs[4], prob, 'pred')


>>>>>>> a5841643530ed1859b3b894843649e2c1f36074e

        if not save_plots:
            plt.show()
        else:
            output_path = ROOT_PATH / 'validation' / 'plots' / config_name
            output_path.mkdir(exist_ok=True)
            file = output_path / f'validation_{config_name}_{i:010d}_{j:010d}.png'
            plt.savefig(file, dpi=300, bbox_inches='tight')
        plt.close()

        if i + 1 == n:
            break


if __name__ == '__main__':
<<<<<<< HEAD
    # show_predictions('impervious_soft', 100, 'validation', 50, save_plots=False)
    show_probabilities('impervious_soft', 100, 'validation', 50, 'imp', save_plots=False)
=======
    show_probabilities('soft_rmse_adamw', 200, 'validation', 50, 'imp', save_plots=True)
>>>>>>> a5841643530ed1859b3b894843649e2c1f36074e
