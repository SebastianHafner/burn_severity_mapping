import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import torch


def compose_transformations(cfg):
    transformations = []

    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        transformations.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    if cfg.AUGMENTATION.CROP_TYPE == 'importance':
        transformations.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if cfg.AUGMENTATION.COLOR:
        transformations.append(ColorShift())

    if cfg.AUGMENTATION.GAMMA:
        transformations.append(GammaCorrection())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, sample: tuple):
        img, label = sample
        img_tensor = TF.to_tensor(img)
        label_tensor = TF.to_tensor(label)
        return img_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, sample):
        img, label = sample
        h_flip = np.random.choice([True, False])
        v_flip = np.random.choice([True, False])

        if h_flip:
            img = np.flip(img, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        if v_flip:
            img = np.flip(img, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        return img, label


class RandomRotate(object):
    def __call__(self, args):
        img, label = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img, label


# Performs uniform cropping on images
# TODO: modifify to only img and label
class UniformCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, img1: np.ndarray, img2: np.ndarray, label: np.ndarray):
        height, width, _ = label.shape
        crop_limit_x = width - self.crop_size
        crop_limit_y = height - self.crop_size
        x = np.random.randint(0, crop_limit_x)
        y = np.random.randint(0, crop_limit_y)

        img1_crop = img1[y:y+self.crop_size, x:x+self.crop_size, ]
        img2_crop = img2[y:y + self.crop_size, x:x + self.crop_size, ]
        label_crop = label[y:y+self.crop_size, x:x+self.crop_size, ]
        return img1_crop, img2_crop, label_crop

    def __call__(self, sample: tuple):
        img1, img2, label = sample
        img1, img2, label = self.random_crop(img1, img2, label)
        return img1, img2, label


# TODO: modify to only label and img
class ImportanceRandomCrop(UniformCrop):
    def __call__(self, sample):
        img1, img2, label = sample

        sample_size = 20
        balancing_factor = 5

        random_crops = [self.random_crop(img1, img2, label) for _ in range(sample_size)]
        crop_weights = np.array([crop_label.sum() for _, _, crop_label in random_crops]) + balancing_factor
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(sample_size, p=crop_weights)
        img1, img2, label = random_crops[sample_idx]

        return img1, img2, label


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        img, label = args
        factors = np.random.uniform(self.min_factor, self.max_factor, img.shape[-1])
        img_rescaled = np.clip(img * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_rescaled, label


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, args):
        img, label = args
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, img.shape[-1])
        img_gamma_corrected = np.clip(np.power(img,gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return img_gamma_corrected, label


