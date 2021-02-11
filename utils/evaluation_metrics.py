import torch

import numpy as np


class MultiThresholdMetric(object):
    def __init__(self, threshold):

        # FIXME Does not operate properly

        '''
        Takes in rasterized and batched images
        :param y_true: [B, H, W]
        :param y_pred: [B, C, H, W]
        :param threshold: [Thresh]
        '''

        self._thresholds = threshold[ :, None, None, None, None] # [Tresh, B, C, H, W]
        self._data_dims = (-1, -2, -3, -4) # For a B/W image, it should be [Thresh, B, C, H, W],

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def _normalize_dimensions(self):
        ''' Converts y_truth, y_label and threshold to [B, Thres, C, H, W]'''
        # Naively assume that all of existing shapes of tensors, we transform [B, H, W] -> [B, Thresh, C, H, W]
        self._thresholds = self._thresholds[ :, None, None, None, None] # [Tresh, B, C, H, W]
        # self._y_pred = self._y_pred[None, ...]  # [B, Thresh, C, ...]
        # self._y_true = self._y_true[None,:, None, ...] # [Thresh, B,  C, ...]

    def add_sample(self, y_true:torch.Tensor, y_pred):
        y_true = y_true.bool()[None,...] # [Thresh, B,  C, ...]
        y_pred = y_pred[None, ...]  # [Thresh, B, C, ...]
        y_pred_offset = (y_pred - self._thresholds + 0.5).round().bool()

        self.TP += (y_true & y_pred_offset).sum(dim=self._data_dims).float()
        self.TN += (~y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FP += (y_true & ~y_pred_offset).sum(dim=self._data_dims).float()
        self.FN += (~y_true & y_pred_offset).sum(dim=self._data_dims).float()

    @property
    def precision(self):
        if hasattr(self, '_precision'):
            '''precision previously computed'''
            return self._precision

        denom = (self.TP + self.FP).clamp(10e-05)
        self._precision = self.TP / denom
        return self._precision

    @property
    def recall(self):
        if hasattr(self, '_recall'):
            '''recall previously computed'''
            return self._recall

        denom = (self.TP + self.FN).clamp(10e-05)
        self._recall = self.TP / denom
        return self._recall

    def compute_basic_metrics(self):
        '''
        Computes False Negative Rate and False Positive rate
        :return:
        '''

        false_pos_rate = self.FP/(self.FP + self.TN)
        false_neg_rate = self.FN / (self.FN + self.TP)

        return false_pos_rate, false_neg_rate

    def compute_f1(self):
        denom = (self.precision + self.recall).clamp(10e-05)
        return 2 * self.precision * self.recall / denom


def true_pos(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * torch.round(y_pred), dim=dim)


def false_pos(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum(y_true * (1. - torch.round(y_pred)), dim=dim)


def false_neg(y_true: torch.Tensor, y_pred: torch.Tensor, dim=0):
    return torch.sum((1. - y_true) * torch.round(y_pred), dim=dim)


def precision(y_true: torch.Tensor, y_pred: torch.Tensor, dim):
    denominator = (true_pos(y_true, y_pred, dim) + false_pos(y_true, y_pred, dim))
    denominator = torch.clamp(denominator, 10e-05)
    return true_pos(y_true, y_pred, dim) / denominator


def recall(y_true: torch.Tensor, y_pred: torch.Tensor, dim):
    denominator = (true_pos(y_true, y_pred, dim) + false_neg(y_true, y_pred, dim))
    denominator = torch.clamp(denominator, 10e-05)
    return true_pos(y_true, y_pred, dim) / denominator


def f1_score(gts: torch.Tensor, preds: torch.Tensor, dim=(-1, -2)):
    gts = gts.float()
    preds = preds.float()

    with torch.no_grad():
        recall_val = recall(gts, preds, dim)
        precision_val = precision(gts, preds, dim)
        denom = torch.clamp( (recall_val + precision_val), 10e-5)

        f1 = 2. * recall_val * precision_val / denom

    return f1


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
