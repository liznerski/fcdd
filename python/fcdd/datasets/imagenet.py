import os
import os.path as pt
import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.online_supervisor import OnlineSupervisor
from fcdd.datasets.outlier_exposure.imagenet import MyImageFolder
from fcdd.datasets.preprocessing import get_target_label_idx
from fcdd.util.logging import Logger
from torch.utils.data import Subset
from torchvision.datasets.imagenet import META_FILE, parse_train_archive, parse_val_archive
from torchvision.datasets.imagenet import verify_str_arg, load_meta_file, check_integrity, parse_devkit_archive
from torchvision.transforms.functional import to_tensor, to_pil_image

ROOT = pt.join(pt.dirname(__file__), '..')


class ADImageNet(TorchvisionDataset):
    ad_classes = ['acorn', 'airliner', 'ambulance', 'American alligator', 'banjo', 'barn', 'bikini', 'digital clock',
                  'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand piano', 'hotdog', 'hourglass', 'manhole cover',
                  'mosque', 'nail', 'parking meter', 'pillow', 'revolver', 'dial telephone', 'schooner',
                  'snowmobile', 'soccer ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
    base_folder = 'imagenet'

    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool, logger: Logger = None):
        """
        AD dataset for ImageNet. Following Hendrycks et al. (https://arxiv.org/abs/1812.04606) this AD dataset
        is limited to 30 of the 1000 classes of Imagenet (see :attr:`ADImageNet.ad_classes`).
        :param root: root directory where data is found or is to be downloaded to
        :param normal_class: the class considered nominal
        :param preproc: the kind of preprocessing pipeline
        :param nominal_label: the label that marks nominal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the nominal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode)
        :param online_supervision: whether to sample anomalies online in each epoch,
            or offline before training (same for all epochs in this case)
        :param logger: logger
        """
        assert online_supervision, 'ImageNet artificial anomaly generation needs to be online'
        assert supervise_mode in ['unsupervised', 'other', 'noise'], \
            'Noise mode "malformed_normal" is not supported for ImageNet because nominal images are loaded ' \
            'only if not replaced by some artificial anomaly (to speedup data preprocessing).'

        root = pt.join(root, self.base_folder)
        super().__init__(root, logger=logger)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (3, 224, 224)
        self.raw_shape = (3, 256, 256)
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 30))
        self.outlier_classes.remove(normal_class)
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        if self.nominal_label != 0:
            self.logprint('Swapping labels, i.e. anomalies are 0 and nominals are 1.')

        # mean and std of original pictures per class
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # different types of preprocessing pipelines, here just choose whether to use augmentations
        if preproc in ['', None, 'default', 'none']:
            test_transform = transform = transforms.Compose([
                transforms.Resize((self.shape[-2], self.shape[-1])),
                transforms.ToTensor(),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['aug1']:
            test_transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-1]),
                transforms.CenterCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-1]),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                transforms.Normalize(mean, std)
            ])
        else:
            raise ValueError('Preprocessing pipeline {} is not known.'.format(preproc))

        target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )
        if supervise_mode not in ['unsupervised', 'other']:
            all_transform = OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit)
        else:
            all_transform = None

        train_set = MyImageNet(
            self.root, supervise_mode, self.raw_shape, split='train', normal_classes=self.normal_classes,
            transform=transform, target_transform=target_transform, all_transform=all_transform, logger=logger
        )
        self.train_ad_classes_idx = train_set.get_class_idx(self.ad_classes)
        train_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            self.train_ad_classes_idx.index(t) if t in self.train_ad_classes_idx else np.nan for t in train_set.targets
        ]
        self._generate_artificial_anomalies_train_set(
            'unsupervised', noise_mode, oe_limit, train_set, normal_class,  # gets rid of true anomalous samples
        )
        self._test_set = MyImageNet(
            self.root, supervise_mode, self.raw_shape, split='val', normal_classes=self.normal_classes,
            transform=test_transform, target_transform=target_transform, logger=logger
        )
        self.test_ad_classes_idx = self._test_set.get_class_idx(self.ad_classes)
        self._test_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            self.test_ad_classes_idx.index(t) if t in self.test_ad_classes_idx else np.nan
            for t in self._test_set.targets
        ]
        self._test_set = Subset(
            self._test_set,
            get_target_label_idx(np.asarray(self._test_set.targets), list(range(len(self.ad_classes))))
        )
        self._test_set.fixed_random_order = MyImageNet.fixed_random_order


class PathsMetaFileImageNet(MyImageFolder):
    """
    Reimplements the basic functionality of the torch ImageNet dataset, i.e. preparation of
    a variable containing image paths, targets, etc.
    Does not yet implement get_item.
    """
    def __init__(self, root, split='train', **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super().__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def parse_archives(self):
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


class MyImageNet(PathsMetaFileImageNet):
    """ ImageNet torch dataset extention, s.t. target_transform and online supervisor is applied """
    # s = len([t for t in self.test_loader.dataset.dataset.targets if not np.isnan(t)])
    # order = np.random.choice(list(range(s)), replace=False, size=s)
    fixed_random_order = np.load(pt.join(ROOT, 'datasets', 'confs', 'imagenet30_test_random_order.npy'))

    def __init__(self, root: str, supervise_mode: str, raw_shape: Tuple[int, int, int],
                 transform=None, target_transform=None,
                 normal_classes=None, all_transform=None, split='train', logger=None):
        super().__init__(root, split, transform=transform, target_transform=target_transform, logger=logger)
        self.normal_classes = normal_classes
        self.all_transform = all_transform
        self.split = split
        self.supervise_mode = supervise_mode
        self.raw_shape = raw_shape

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.all_transform is not None:
            replace = random.random() < 0.5
            if replace:
                if self.supervise_mode not in ['malformed_normal', 'malformed_normal_gt']:
                    img, _, target = self.all_transform(
                        torch.empty(self.raw_shape), None, target, replace=replace
                    )
                else:
                    path, _ = self.samples[index]
                    img = to_tensor(self.loader(path)).mul(255).byte()
                    img, _, target = self.all_transform(img, None, target, replace=replace)
                img = to_pil_image(img)
            else:
                path, _ = self.samples[index]
                img = self.loader(path)

        else:
            path, _ = self.samples[index]
            img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_class_idx(self, classes: List[str]):
        return [self.class_to_idx[c] for c in classes]
