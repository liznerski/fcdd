import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
import numpy as np
import random
from abc import abstractmethod


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def local_contrast_normalization(x: torch.tensor, scale='l2'):
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


def get_min_max_mean_std(loader, ch=3):
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x), all_y.append(y)
    x, y = torch.cat(all_x), torch.cat(all_y)
    min, max, mean, std = [], [], [], []
    for l in sorted(set(y.tolist())):
        min.append(x[y == l].transpose(0, 1).reshape(ch, -1).min(1)[0].tolist())
        max.append(x[y == l].transpose(0, 1).reshape(ch, -1).max(1)[0].tolist())
        mean.append(x[y == l].transpose(0, 1).reshape(ch, -1).mean(1).tolist())
        std.append(x[y == l].transpose(0, 1).reshape(ch, -1).std(1).tolist())
    return min, max, mean, std


def get_statistics(self, TorchDatasetClass, numClasses, shape):
    stats = []
    stats_gcn = []
    for c in range(numClasses):
        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = shape
        self.normal_classes = tuple([c])
        self.outlier_classes = list(range(0, numClasses))
        self.outlier_classes.remove(c)
        test_transform = transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        train_set = TorchDatasetClass(root=self.root, split='train', download=True,
                                      transform=transform, target_transform=target_transform)

        self._generate_artificial_anomalies_train_set('unsupervised', {}, train_set, c)

        self._test_set = TorchDatasetClass(root=self.root, split='test', download=True,
                                           transform=test_transform, target_transform=target_transform)

        mi, ma, mean, std = get_min_max_mean_std(self.loaders(100)[0], ch=self.shape[0])
        stats.append(tuple([mi, ma, mean, std]))

        test_transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
        ])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        train_set = TorchDatasetClass(root=self.root, split='train', download=True,
                                      transform=transform, target_transform=target_transform)

        self._generate_artificial_anomalies_train_set('unsupervised', {}, train_set, c)

        self._test_set = TorchDatasetClass(root=self.root, split='test', download=True,
                                           transform=test_transform, target_transform=target_transform)
        mi, ma, mean, std = get_min_max_mean_std(self.loaders(100)[0], ch=self.shape[0])
        stats_gcn.append(tuple([mi, ma, mean, std]))
    return stats, stats_gcn


class MultiCompose(transforms.Compose):
    def __call__(self, imgs):
        for t in self.transforms:
            imgs = list(imgs)
            imgs = self.multi_apply(imgs, t)
        return imgs

    def multi_apply(self, imgs, t):
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
            imgs[:-1] = self.multi_apply(imgs[:-1], t_picked)
        elif isinstance(t, MultiTransform):
            assert t.n == len(imgs)
            imgs = t(*imgs)
        elif isinstance(t, transforms.RandomChoice):
            t_picked = random.choice(t.transforms)
            imgs = self.multi_apply(imgs, t_picked)
        elif isinstance(t, MultiCompose):
            imgs = t(imgs)
        else:
            raise NotImplementedError('There is no multi compose version of {} yet.'.format(t.__class__))
        return imgs


class MultiTransform(object):
    n = 0  # amount of expected inputs
    pass


class ImgGTTargetTransform(MultiTransform):
    n = 3
    @abstractmethod
    def __call__(self, img, gt, target):
        return img, gt, target


class ImgGtTransform(MultiTransform):
    n = 2
    @abstractmethod
    def __call__(self, img, gt):
        return img, gt


class MajorityVoteGtMap(ImgGtTransform):
    def __call__(self, img, gt):
        return img, gt.fill_(1 if gt[gt == 1].sum() > gt[gt == 0].sum() else 0)


class LabelConditioner(ImgGTTargetTransform):
    def __init__(self, conds, t1, t2):
        self.conds = conds
        self.t1 = t1
        self.t2 = t2

    def __call__(self, img, gt, target):
        if target in self.conds:
            return self.t1
        else:
            return self.t2


class ImgTransformWrap(ImgGtTransform):
    def __init__(self, t):
        self.t = t

    def __call__(self, img, gt):
        return self.t(img), gt


class BlackCenter(object):
    def __init__(self, percentage=0.5, inverse=False):
        self.percentage = percentage
        self.inverse = inverse

    def __call__(self, img):
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


