import torch
from utils import evaluation, datasets, network
from experiment_manager.config import config
from pathlib import Path
import numpy as np

ROOT_PATH = Path('/storage/shafner/burn_severity_mapping')
CONFIG_PATH = Path('/home/shafner/burn_severity_mapping/configs')
NETWORK_PATH = Path('/storage/shafner/burn_severity_mapping/networks')


def run_validation(config_name: str, site: str):

    # loading cfg and network
    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    net = network.load_net(cfg, NETWORK_PATH / f'{config_name}_{cfg.CHECKPOINT}.pkl')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    measurer = evaluation.MultiClassEvaluation(cfg.MODEL.OUT_CHANNELS)

    dataset = datasets.TrainingDataset(cfg, 'validation', no_augmentation=True, site=site)

    def evaluation_callback(x, y, z):
        # x img y label z logits
        measurer.add_sample(z, y)

    evaluation.inference_loop(net, cfg, device, dataset, evaluation_callback, num_workers=8)

    print(f'Computing overall accuracy', end=' ', flush=True)

    # total assessment
    oacc = measurer.overall_accuracy()
    print(f'{oacc:.2f}', flush=True)

    # confusion matrix
    print(measurer.confusion_matrix())

    # per-class assessment
    classes = cfg.DATASET.CLASSES
    for i, class_ in enumerate(classes):

        f1_score, precision, recall = measurer.class_evaluation(i)
        print(f'{class_}: f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')


def validation_stats(config_name: str, site: str):

    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = datasets.TrainingDataset(cfg, 'validation', no_augmentation=True, site=site)

    n_classes = cfg.MODEL.OUT_CHANNELS
    n_class_pixels = np.zeros(n_classes)
    bins = np.arange(-0.5, n_classes, 1)

    for index in range(len(dataset)):
        patch = dataset.samples[index]
        label = dataset.get_label(patch['site'], patch['x'], patch['y'])
        hist_sample, _ = np.histogram(label, bins=bins)
        n_class_pixels += hist_sample

    print(n_class_pixels)
    print(n_class_pixels / np.sum(n_class_pixels) * 100)


def validate_dnbr(config_name: str, site: str):

    cfg = config.load_cfg(CONFIG_PATH / f'{config_name}.yaml')
    dataset = datasets.TrainingDataset(cfg, 'validation', no_augmentation=True, site=site)
    measurer = evaluation.MultiClassEvaluation(cfg.MODEL.OUT_CHANNELS)

    for index in range(len(dataset.samples)):

        sample = dataset.samples[index]
        x, y = sample['x'], sample['y']

        dnbr = dataset.get_auxilliary_data('dnbr', site, x, y)
        dnbr_thresholded = dataset.threshold(dnbr, cfg.DATASET.THRESHOLDS)

        label = dataset.get_label(site, x, y)

        measurer.predictions.extend(dnbr_thresholded.flatten())
        measurer.labels.extend(label.flatten())


    print(f'Computing overall accuracy', end=' ', flush=True)
    # total assessment
    oacc = measurer.overall_accuracy()
    print(f'{oacc:.2f}', flush=True)

    # confusion matrix
    print(measurer.confusion_matrix())

    # per-class assessment
    classes = cfg.DATASET.CLASSES
    for i, class_ in enumerate(classes):

        f1_score, precision, recall = measurer.class_evaluation(i)
        print(f'{class_}: f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')


if __name__ == '__main__':
    config_name = 'dnbr_validation_canada'
    site = 'elephanthill'
    # run_validation(config_name, site)
    # validate_dnbr(config_name, site)
    validation_stats(config_name, site)
