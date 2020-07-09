import os
import os.path as pt
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.online_superviser import OnlineSuperviser
from fcdd.datasets.outlier_exposure.imagenet import MyImageFolder
from fcdd.datasets.preprocessing import get_target_label_idx
from torch.utils.data import Subset
from torchvision.datasets.imagenet import META_FILE, parse_train_archive, parse_val_archive
from torchvision.datasets.imagenet import verify_str_arg, load_meta_file, check_integrity, parse_devkit_archive

ROOT = pt.join(pt.dirname(__file__), '..')


class ADImageNet(TorchvisionDataset):
    ad_classes = ['acorn', 'airliner', 'ambulance', 'American alligator', 'banjo', 'barn', 'bikini', 'digital clock',
                  'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand piano', 'hotdog', 'hourglass', 'manhole cover',
                  'mosque', 'nail', 'parking meter', 'pillow', 'revolver', 'dial telephone', 'schooner',
                  'snowmobile', 'soccer ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
    base_folder = 'imagenet'

    def __init__(self, root: str, normal_class=1, preproc='aug1',
                 supervise_mode='unsupervised', supervise_params=None, logger=None):
        assert supervise_params.get('online', True), 'ImageNet artificial anomaly generation needs to be online'
        assert supervise_mode in ['unsupervised', 'other', 'noise'], \
            'malformed_normal is not supported, because nominal image is only loaded ' \
            'if not replaced by anomaly to speedup data preprocessing'

        root = pt.join(root, self.base_folder)
        super().__init__(root, logger=logger)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (3, 224, 224)
        self.raw_shape = (3, 256, 256)
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 30))
        self.outlier_classes.remove(normal_class)
        assert supervise_params.get('nominal_label', 0) in [0, 1]
        self.nominal_label = supervise_params.get('nominal_label', 0)
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        if self.nominal_label != 0:
            self.logprint('Swapping labels, i.e. anomalies are 0 and nominals are 1.')

        # mean and std of original pictures per class
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # different types of preprocessing pipelines, here just choose whether to use augmentations
        if preproc in ['', None, 'default', 'none']:
            test_transform = transform = transforms.Compose([
                transforms.Resize(self.shape[-1]),
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
            raise ValueError('Preprocessing set {} is not known.'.format(preproc))

        target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )
        if supervise_mode not in ['unsupervised', 'other']:
            all_transform = OnlineSuperviser(self, supervise_mode, supervise_params)
        else:
            all_transform = None

        train_set = MyImageNet(
            root=self.root, split='train', normal_classes=self.normal_classes,
            transform=transform, target_transform=target_transform, all_transform=all_transform, logger=logger
        )
        self.train_ad_classes_idx = train_set.get_class_idx(self.ad_classes)
        train_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            self.train_ad_classes_idx.index(t) if t in self.train_ad_classes_idx else np.nan for t in train_set.targets
        ]
        self._generate_artificial_anomalies_train_set(
            'unsupervised', supervise_params, train_set, normal_class,
        )
        self._test_set = MyImageNet(
            root=self.root, split='val', normal_classes=self.normal_classes,
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
    # s = len([t for t in self.test_loader.dataset.dataset.targets if not np.isnan(t)])
    # order = np.random.choice(list(range(s)), replace=False, size=s)
    fixed_random_order = np.load(pt.join(ROOT, 'datasets', 'confs', 'imagenet30_test_random_order.npy'))

    def __init__(self, root, transform=None, target_transform=None,
                 normal_classes=None, all_transform=None, split='train', logger=None):
        super().__init__(root, split, transform=transform, target_transform=target_transform, logger=logger)
        self.normal_classes = normal_classes
        self.all_transform = all_transform
        self.split = split

    def __getitem__(self, index):
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.all_transform is not None:
            replace = random.random() < 0.5
            if replace:
                img, _, target = self.all_transform(None, None, target, replace=replace)
                img = transforms.ToPILImage()(img)
            else:
                path, _ = self.samples[index]
                img = self.loader(path)

        else:
            path, _ = self.samples[index]
            img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_class_idx(self, classes):
        return [self.class_to_idx[c] for c in classes]
