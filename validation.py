import torch
from utils import evaluation, datasets, network
from experiment_manager.config import config
from pathlib import Path

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

    dataset = datasets.TrainingDataset(cfg, 'validation', no_augmentation=True)

    def evaluation_callback(x, y, z):
        # x img y label z logits
        measurer.add_sample(z, y)

    evaluation.inference_loop(net, cfg, device, dataset, evaluation_callback, num_workers=8)

    print(f'Computing overall accuracy', end=' ', flush=True)

    # total assessment
    oacc = measurer.overall_accuracy()
    print(f'{oacc:.2f}', flush=True)

    # per-class assessment
    classes = cfg.DATASET.CLASSES
    for i, class_ in enumerate(classes):

        f1_score, precision, recall = measurer.class_evaluation(i)
        print(f'{class_}: f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')


if __name__ == '__main__':
    config_name = 'baseline_sar'
    site = 'elephanthill'
    run_validation(config_name, site)
