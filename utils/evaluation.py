import torch
from torch.utils import data as torch_data
from utils import datasets
import numpy as np
import wandb


def model_eval(net, cfg, device, run_type, epoch, step, max_samples: int = 100):
    measurer = MultiClassEvaluation(cfg.MODEL.OUT_CHANNELS)

    dataset = datasets.TrainingDataset(cfg, run_type, no_augmentation=True)

    def evaluation_callback(x, y, z):
        # x img y label z logits
        measurer.add_sample(z, y)

    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    inference_loop(net, cfg, device, dataset, evaluation_callback, max_samples=max_samples, num_workers=num_workers)

    print(f'Computing {run_type} overall accuracy', end=' ', flush=True)

    # total assessment
    oacc = measurer.overall_accuracy()
    print(f'{oacc:.2f}', flush=True)
    if not cfg.DEBUG:
        wandb.log({
            f'{run_type} oacc': oacc,
            'step': step,
            'epoch': epoch,
        })

    # per-class assessment
    classes = cfg.DATASET.CLASSES
    for i, class_ in enumerate(classes):

        # n_pred, n_true = measurer.class_statistics(i)
        # print(f'{class_}: n predictions {n_pred} - n labels {n_true}')

        f1_score, precision, recall = measurer.class_evaluation(i)
        print(f'{class_}: f1 score {f1_score:.3f} - precision {precision:.3f} - recall {recall:.3f}')
        if not cfg.DEBUG:
            wandb.log({
                f'{run_type} {class_} f1_score': f1_score,
                f'{run_type} {class_} precision': precision,
                f'{run_type} {class_} recall': recall,
                'step': step,
                'epoch': epoch,
            })


def inference_loop(net, cfg, device, dataset, callback, max_samples=None, num_workers=0):
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': num_workers,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    net.to(device)
    max_samples = len(dataset) if max_samples is None else max_samples

    counter = 0
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):
            img = batch['img'].to(device)
            label = batch['label'].to(device)

            logits = net(img)

            callback(img, label, logits)

            counter += 1
            if counter == max_samples or cfg.DEBUG:
                break


class MultiClassEvaluation(object):
    def __init__(self, n_classes: int, class_names: list = None):
        self.n_classes = n_classes
        self.class_names = class_names
        self.predictions = []
        self.labels = []

    def add_sample(self, logits: torch.tensor, label: torch.tensor):

        sm = torch.nn.Softmax(dim=1)
        prob = sm(logits)
        pred = torch.argmax(prob, dim=1)
        pred = pred.float().detach().cpu().numpy()
        label = label.float().detach().cpu().numpy()

        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())

    def reset(self):
        self.predictions = []
        self.labels = []

    def overall_accuracy(self) -> float:
        acc = np.array(self.predictions) == np.array(self.labels)
        return float(100 * np.sum(acc) / np.size(acc))

    def class_evaluation(self, class_: int) -> tuple:
        y_pred = np.array(self.predictions) == class_
        y_true = np.array(self.labels) == class_
        tp = np.sum(np.logical_and(y_true, y_pred))
        fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
        fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
        prec = tp / (tp + fp) if tp + fp != 0 else 0
        rec = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if prec + rec != 0 else 0
        return f1, prec, rec

    def class_statistics(self, class_: int) -> tuple:
        y_pred = np.array(self.predictions) == class_
        y_true = np.array(self.labels) == class_
        n_pred = np.sum(y_pred)
        n_true = np.sum(y_true)
        return n_pred, n_true

    def confusion_matrix(self) -> np.ndarray:
        cm = np.zeros(self.n_classes, self.n_classes)
        for pred, label in zip(self.predictions, self.labels):
            cm[label, pred] += 1
        return cm


if __name__ == '__main__':
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    classes = ['Cat', 'Fish', 'Hen']
    measurer = MultiClassEvaluation(len(classes), classes)
    predictions = [0] * 13 + [1] * 3 + [2] * 9
    labels = [0] * 4 + [1] * 6 + [2] * 3 + [0] + [1, 1] + [0] + [1, 1] + [2] * 6
    measurer.predictions = predictions
    measurer.labels = labels
    for i, class_ in enumerate(classes):
        print(class_)
        f1, prec, rec = measurer.class_evaluation(i)
        print(f'{f1:.3f}, {prec:.3f}, {rec:.3f}')
