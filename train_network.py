# general modules
import sys
import os
import numpy as np
from pathlib import Path

# learning framework
from torch.utils import data as torch_data

# config for experiments
from experiment_manager import args
from experiment_manager.config import config

# custom stuff
from utils import datasets, network
from utils.loss_functions import *
from utils.evaluation import model_eval

# logging
import wandb


def setup(args):
    cfg = config.new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file
    return cfg


def train(net, cfg):

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    net.to(device)

    # reset the generators
    dataset = datasets.TrainingDataset(cfg, 'train')
    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    if cfg.AUGMENTATION.OVERSAMPLING:
        dataloader_kwargs['sampler'] = dataset.sampler()
        dataloader_kwargs['shuffle'] = False

    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
    steps_per_epoch = len(dataloader)

    def get_optimizer(net, optim, lr, wd):
        if optim == 'adam':
            return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        elif optim == 'adamw':
            return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        else:
            return torch.optim.Adam(net.parameters())
    learning_rate = cfg.TRAINER.LR
    optimizer = get_optimizer(net, cfg.TRAINER.OPTIMIZER, learning_rate, cfg.TRAINER.WEIGHT_DECAY)

    # loss function
    loss_type = cfg.MODEL.LOSS_TYPE
    if loss_type == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == 'SoftCrossEntropyLoss':
        criterion = soft_cross_entropy_loss
    elif loss_type == 'WeightedCrossEntropyLoss':
        class_weights = dataset.class_weights()
        class_weights = torch.tensor(class_weights).to(device).float()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'RMSE':
        criterion = root_mean_square_error_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()

    save_path = Path(cfg.OUTPUT_BASE_DIR) / cfg.NAME
    save_path.mkdir(exist_ok=True)

    global_step = epoch_float = 0
    epochs = cfg.TRAINER.EPOCHS
    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch} / {epochs}')

        loss_tracker = 0
        net.train()
        if epoch != 0 and (epoch % cfg.TRAINER.LR_DECAY_INTERVAL == 0):
            print('learning rate decay')
            learning_rate = learning_rate / cfg.TRAINER.LR_DECAY_FACTOR
            optimizer = get_optimizer(net, cfg.TRAINER.OPTIMIZER, learning_rate, cfg.TRAINER.WEIGHT_DECAY)

        for i, batch in enumerate(dataloader):

            img = batch['img'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            logits = net(img)

            loss = criterion(logits, label.squeeze().long())
            loss_tracker += loss.item()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if cfg.DEBUG:
                break

        print(f'loss: {loss_tracker:.3f}')
        if not cfg.DEBUG:
            assert (epoch == epoch_float)
        model_eval(net, cfg, device, 'train', epoch_float, global_step)
        model_eval(net, cfg, device, 'validation', epoch_float, global_step)
        # end of epoch

        if epoch in cfg.SAVE_CHECKPOINTS and not cfg.DEBUG:
            print(f'saving network', flush=True)
            net_file = Path(cfg.OUTPUT_BASE_DIR) / f'{cfg.NAME}_{epoch}.pkl'
            torch.save(net.state_dict(), net_file)


if __name__ == '__main__':

    # setting up config based on parsed argument
    parser = args.default_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = setup(args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # loading network
    net = network.UNet(cfg)

    # tracking land with w&b
    if not cfg.DEBUG:
        wandb.init(
            name=cfg.NAME,
            project='burn_severity_mapping',
            tags=['run', 'wildfire', 'burn severity' ]
        )

    try:
        train(net, cfg)
    except KeyboardInterrupt:
        print('Training terminated')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
