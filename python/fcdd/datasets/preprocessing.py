import random
from abc import abstractmethod
from typing import Callable, List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def get_target_label_idx(labels: np.ndarray, targets: np.ndarray):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def local_contrast_normalization(x: torch.tensor, scale: str = 'l2'):
    """
    Apply local contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale if x_scale != 0 else 1

    return x


class MultiCompose(transforms.Compose):
    """
    Like transforms.Compose, but applies all transformations to a multitude of variables, instead of just one.
    More importantly, for random transformations (like RandomCrop), applies the same choice of transformation, i.e.
    e.g. the same crop for all variables.
    """
    def __call__(self, imgs: List):
        for t in self.transforms:
            imgs = list(imgs)
            imgs = self.__multi_apply(imgs, t)
        return imgs

    def __multi_apply(self, imgs: List, t: Callable):
        if isinstance(t, transforms.RandomCrop):
            for idx, img in enumerate(imgs):
                if t.padding is not None and t.padding > 0:
                    img = TF.pad(img, t.padding, t.fill, t.padding_mode) if img is not None else img
                if t.pad_if_needed and img.size[0] < t.size[1]:
                    img = TF.pad(img, (t.size[1] - img.size[0], 0), t.fill, t.padding_mode) if img is not None else img
                if t.pad_if_needed and img.size[1] < t.size[0]:
                    img = TF.pad(img, (0, t.size[0] - img.size[1]), t.fill, t.padding_mode) if img is not None else img
                imgs[idx] = img
            i, j, h, w = t.get_params(imgs[0], output_size=t.size)
            for idx, img in enumerate(imgs):
                imgs[idx] = TF.crop(img, i, j, h, w) if img is not None else img
        elif isinstance(t, transforms.RandomHorizontalFlip):
            if random.random() > 0.5:
                for idx, img in enumerate(imgs):
                    imgs[idx] = TF.hflip(img)
        elif isinstance(t, transforms.RandomVerticalFlip):
            if random.random() > 0.5:
                for idx, img in enumerate(imgs):
                    imgs[idx] = TF.vflip(img)
        elif isinstance(t, transforms.ToTensor):
            for idx, img in enumerate(imgs):
                imgs[idx] = TF.to_tensor(img) if img is not None else None
        elif isinstance(
                t, (transforms.Resize, transforms.Lambda, transforms.ToPILImage, transforms.ToTensor, BlackCenter)
        ):
            for idx, img in enumerate(imgs):
                imgs[idx] = t(img) if img is not None else None
        elif isinstance(t, LabelConditioner):
            assert t.n == len(imgs)
            t_picked = t(*imgs)
            imgs[:-1] = self.__multi_apply(imgs[:-1], t_picked)
        elif isinstance(t, MultiTransform):
            assert t.n == len(imgs)
            imgs = t(*imgs)
        elif isinstance(t, transforms.RandomChoice):
            t_picked = random.choice(t.transforms)
            imgs = self.__multi_apply(imgs, t_picked)
        elif isinstance(t, MultiCompose):
            imgs = t(imgs)
        else:
            raise NotImplementedError('There is no multi compose version of {} yet.'.format(t.__class__))
        return imgs


class MultiTransform(object):
    """ Class to mark a transform operation that expects multiple inputs """
    n = 0  # amount of expected inputs
    pass


class ImgGTTargetTransform(MultiTransform):
    """ Class to mark a transform operation that expects three inputs: (image, ground-truth map, label) """
    n = 3
    @abstractmethod
    def __call__(self, img, gt, target):
        return img, gt, target


class ImgGtTransform(MultiTransform):
    """ Class to mark a transform operation that expects two inputs: (image, ground-truth map) """
    n = 2
    @abstractmethod
    def __call__(self, img, gt):
        return img, gt


class LabelConditioner(ImgGTTargetTransform):
    def __init__(self, conds: List[int], t1: Callable, t2: Callable):
        """
        Applies transformation t1 if the encountered label is in conds, otherwise applies transformation t2.
        :param conds: list of labels
        :param t1: some transformation
        :param t2: some other transformation
        """
        self.conds = conds
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img, gt, target):
        if target in self.conds:
            return self.t1
        else:
            return self.t2


class ImgTransformWrap(ImgGtTransform):
    """ Wrapper for some transformation that is used in a MultiCompose, but is only to be applied to the first input """
    def __init__(self, t: Callable):
        self.t = t

    def __call__(self, img, gt):
        return self.t(img), gt


class BlackCenter(object):
    def __init__(self, percentage: float = 0.5, inverse: bool = False):
        """
        Blackens the center of given image, i.e. puts pixel value to zero.
        :param percentage: the percentage of the center in the overall image.
        :param inverse: whether to inverse the operation, i.e. blacken the borders instead.
        """
        self.percentage = percentage
        self.inverse = inverse

    def __call__(self, img: torch.Tensor):
        c, h, w = img.shape
        oh, ow = int((1 - self.percentage) * h * 0.5), int((1 - self.percentage) * w * 0.5)
        if not self.inverse:
            img[:, oh:-oh, ow:-ow] = 0
        else:
            img[:, :oh, :] = 0
            img[:, -oh:, :] = 0
            img[:, :, :ow] = 0
            img[:, :, -ow:] = 0
        return img


