import random
import traceback
from itertools import cycle
from typing import List, Tuple

import numpy as np
import torch
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.outlier_exposure.cifar100 import OECifar100
from fcdd.datasets.outlier_exposure.emnist import OEEMNIST
from fcdd.datasets.outlier_exposure.imagenet import OEImageNet, OEImageNet22k
from fcdd.datasets.outlier_exposure.mvtec import OEMvTec
from fcdd.datasets.preprocessing import ImgGTTargetTransform


class OnlineSupervisor(ImgGTTargetTransform):
    invert_threshold = 0.025

    def __init__(self, ds: TorchvisionDataset, supervise_mode: str, noise_mode: str, oe_limit: int = np.infty,
                 p: float = 0.5, exclude: List[str] = ()):
        """
        This class is used as a Transform parameter for torchvision datasets.
        During training it randomly replaces a sample of the dataset retrieved via the get_item method
        by an artificial anomaly.
        :param ds: some AD dataset for which the OnlineSupervisor is used.
        :param supervise_mode: the type of artificial anomalies to be generated during training.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
            In addition to the offline noise modes, the OnlineSupervisor offers Outlier Exposure with MVTec-AD.
            The oe_limit parameter for MVTec-AD limits the number of different samples per defection type
            (including "good" instances, i.e. nominal ones in the test set).
        :param oe_limit: the number of different Outlier Exposure samples used in case of outlier exposure based noise.
        :param p: the chance to replace a sample from the original dataset during training.
        :param exclude: all class names that are to be excluded in Outlier Exposure datasets.
        """
        self.ds = ds
        self.supervise_mode = supervise_mode
        self.noise_mode = noise_mode
        self.oe_limit = oe_limit
        self.p = p
        self.noise_sampler = None
        if noise_mode == 'imagenet':
            self.noise_sampler = cycle(
                OEImageNet(
                    (1, ) + ds.raw_shape, limit_var=oe_limit, root=ds.root, exclude=exclude
                ).data_loader()
            )
        elif noise_mode == 'imagenet22k':
            self.noise_sampler = cycle(
                OEImageNet22k(
                    (1, ) + ds.raw_shape, limit_var=oe_limit, logger=ds.logger,
                    root=ds.root
                ).data_loader()
            )
        elif noise_mode == 'cifar100':
            self.noise_sampler = cycle(
                OECifar100(
                    (1, ) + ds.raw_shape, limit_var=oe_limit,
                    root=ds.root
                ).data_loader(),
            )
        elif noise_mode == 'emnist':
            self.noise_sampler = cycle(
                OEEMNIST(
                    (1, ) + ds.raw_shape, limit_var=oe_limit,
                    root=ds.root
                ).data_loader()
            )
        elif noise_mode == 'mvtec':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, ds.normal_classes, limit_var=oe_limit,
                    logger=ds.logger, root=ds.root
                ).data_loader()
            )
        elif noise_mode == 'mvtec_gt':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, ds.normal_classes, limit_var=oe_limit,
                    logger=ds.logger, gt=True, root=ds.root
                ).data_loader()
            )

    def __call__(self, img: torch.Tensor, gt: torch.Tensor, target: int,
                 replace: bool = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Based on the probability defined in __init__, replaces (img, gt, target) with an artificial anomaly.
        :param img: some torch tensor image
        :param gt: some ground-truth map (can be None)
        :param target: some label
        :param replace: whether to force or forbid a replacement, ignoring the probability.
            The probability is only considered if replace == None.
        :return: (img, gt, target)
        """
        active = self.supervise_mode not in ['other', 'unsupervised']
        if active and (replace or replace is None and random.random() < self.p):
            supervise_mode = self.supervise_mode
            img = img.unsqueeze(0) if img is not None else img
            # gt value 1 will be put to anom_label in mvtec_bases get_item
            gt = gt.unsqueeze(0).unsqueeze(0).fill_(1).float() if gt is not None else gt
            if self.noise_sampler is None:
                generated_noise = self.ds._generate_noise(
                    self.noise_mode, img.shape
                )
            else:
                try:
                    generated_noise = next(self.noise_sampler)
                except RuntimeError:
                    generated_noise = next(self.noise_sampler)
                    self.ds.logger.warning(
                        'Had to resample in online_supervisor __call__ next(self.noise_sampler) because of {}'
                        .format(traceback.format_exc())
                    )
                if isinstance(generated_noise, (tuple, list)):
                    generated_noise, gt = generated_noise
            if supervise_mode in ['noise']:
                img, gt, target = self.__noise(img, gt, target, self.ds, generated_noise)
            elif supervise_mode in ['malformed_normal']:
                img, gt, target = self.__malformed_normal(
                    img, gt, target, self.ds, generated_noise, invert_threshold=self.invert_threshold
                )
            elif supervise_mode in ['malformed_normal_gt']:
                img, gt, target = self.__malformed_normal(
                    img, gt, target, self.ds, generated_noise, use_gt=True,
                    invert_threshold=self.invert_threshold
                )
            else:
                raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
            img = img.squeeze(0) if img is not None else img
            gt = gt.squeeze(0).squeeze(0) if gt is not None else gt
        return img, gt, target

    def __noise(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset,
                generated_noise: torch.Tensor, use_gt: bool = False):
        if use_gt:
            raise ValueError('No GT mode for pure noise available!')
        anom = generated_noise.clamp(0, 255).byte()
        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied
        return anom, gt, t

    def __malformed_normal(self, img: torch.Tensor, gt: torch.Tensor, target: int, ds: TorchvisionDataset,
                           generated_noise: torch.Tensor, use_gt: bool = False, invert_threshold: float = 0.025):
        assert (img.dim() == 4 or img.dim() == 3) and generated_noise.shape == img.shape
        anom = img.clone()

        # invert noise if difference of malformed and original is less than threshold and inverted difference is higher
        diff = ((anom.int() + generated_noise).clamp(0, 255) - anom.int())
        diff = diff.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()
        diffi = ((anom.int() - generated_noise).clamp(0, 255) - anom.int())
        diffi = diffi.reshape(anom.size(0), -1).sum(1).float().div(np.prod(anom.shape)).abs()
        inv = [i for i, (d, di) in enumerate(zip(diff, diffi)) if d < invert_threshold and di > d]
        generated_noise[inv] = -generated_noise[inv]

        anom = (anom.int() + generated_noise).clamp(0, 255).byte()

        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied

        if use_gt:
            gt = (img != anom).max(1)[0].clone().float()
            gt = gt.unsqueeze(1)  # value 1 will be put to anom_label in mvtec_bases get_item

        return anom, gt, t
