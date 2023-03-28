import random
import os
import os.path as pt
import numpy as np
import torchvision.transforms as transforms
import torch
from typing import Tuple, List
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor, to_pil_image
from fcdd.datasets.bases import TorchvisionDataset, GTSubset
from fcdd.datasets.online_supervisor import OnlineSupervisor
from fcdd.datasets.preprocessing import get_target_label_idx
from fcdd.util.logging import Logger


def extract_custom_classes(datapath: str) -> List[str]:
    dir = os.path.join(datapath, 'custom', 'test')
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    return classes


class ADImageFolderDataset(TorchvisionDataset):
    base_folder = 'custom'
    ovr = False
    gtm = False

    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool,
                 logger: Logger = None):
        """
        This is a general-purpose implementation for custom datasets.
        It expects the data being contained in class folders and distinguishes between
        (1) the one-vs-rest (ovr) approach where one class is considered normal
        and is tested against all other classes being anomalous
        (2) the general approach where each class folder contains a normal data folder and an anomalous data folder.
        The :attr:`ovr` determines this.

        For (1) the data folders have to follow this structure:
        root/custom/train/dog/xxx.png
        root/custom/train/dog/xxy.png
        root/custom/train/dog/xxz.png

        root/custom/train/cat/123.png
        root/custom/train/cat/nsdf3.png
        root/custom/train/cat/asd932_.png

        For (2):
        root/custom/train/hazelnut/normal/xxx.png
        root/custom/train/hazelnut/normal/xxy.png
        root/custom/train/hazelnut/normal/xxz.png
        root/custom/train/hazelnut/anomalous/xxa.png    -- may be used during training for a semi-supervised setting

        root/custom/train/screw/normal/123.png
        root/custom/train/screw/normal/nsdf3.png
        root/custom/train/screw/anomalous/asd932_.png   -- may be used during training for a semi-supervised setting

        The same holds for the test set, where "train" has to be replaced by "test".

        :param root: root directory where data is found.
        :param normal_class: the class considered normal.
        :param preproc: the kind of preprocessing pipeline.
        :param nominal_label: the label that marks normal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the normal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode).
        :param online_supervision: whether to sample anomalies online in each epoch,
            or offline before training (same for all epochs in this case).
        :param logger: logger.
        """
        assert online_supervision, 'Artificial anomaly generation for custom datasets needs to be online'
        self.trainpath = pt.join(root, self.base_folder, 'train')
        self.testpath = pt.join(root, self.base_folder, 'test')
        super().__init__(root, logger=logger)
        self.check_data()

        self.n_classes = 2  # 0: normal, 1: outlier
        self.raw_shape = (3, 248, 248)
        self.shape = (3, 224, 224)  # shape of your data samples in channels x height x width after image preprocessing
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, len(extract_custom_classes(root))))
        self.outlier_classes.remove(normal_class)
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        # precomputed mean and std of your training data
        self.mean, self.std = self.extract_mean_std(self.trainpath, normal_class)

        if preproc in ['', None, 'default', 'none']:
            test_transform = transform = transforms.Compose([
                transforms.Resize((self.shape[-2], self.shape[-1])),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        elif preproc in ['aug1']:
            test_transform = transforms.Compose([
                transforms.Resize((self.raw_shape[-1])),
                transforms.CenterCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-1]),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                transforms.Normalize(self.mean, self.std)
            ])
        #  here you could define other pipelines with augmentations
        else:
            raise ValueError('Preprocessing pipeline {} is not known.'.format(preproc))

        self.target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )
        if supervise_mode not in ['unsupervised', 'other']:
            self.all_transform = OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit)
        else:
            self.all_transform = None

        self._train_set = ImageFolderDataset(
            self.trainpath, supervise_mode, self.raw_shape, self.ovr, self.nominal_label, self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=transform, target_transform=self.target_transform, all_transform=self.all_transform,
        )
        if supervise_mode == 'other':  # (semi)-supervised setting
            self.balance_dataset()
        else:
            self._train_set = Subset(
                self._train_set, np.argwhere(
                    (np.asarray(self._train_set.anomaly_labels) == self.nominal_label) *
                    np.isin(self._train_set.targets, self.normal_classes)
                ).flatten().tolist()
            )

        self._test_set = ImageFolderDataset(
            self.testpath, supervise_mode, self.raw_shape, self.ovr, self.nominal_label, self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=test_transform, target_transform=self.target_transform,
        )
        if not self.ovr:
            self._test_set = Subset(
                self._test_set, get_target_label_idx(self._test_set.targets, np.asarray(self.normal_classes))
            )

    def balance_dataset(self, gtm=False):
        nominal_mask = (np.asarray(self._train_set.anomaly_labels) == self.nominal_label)
        nominal_mask = nominal_mask * np.isin(self._train_set.targets, np.asarray(self.normal_classes))
        anomaly_mask = (np.asarray(self._train_set.anomaly_labels) != self.nominal_label)
        anomaly_mask = anomaly_mask * (1 if self.ovr else np.isin(
            self._train_set.targets, np.asarray(self.normal_classes)
        ))

        if anomaly_mask.sum() == 0:
            return

        CLZ = Subset if not gtm else GTSubset
        self._train_set = CLZ(  # randomly pick n_nominal anomalies for a balanced training set
            self._train_set, np.concatenate([
                np.argwhere(nominal_mask).flatten().tolist(),
                np.random.choice(np.argwhere(anomaly_mask).flatten().tolist(), nominal_mask.sum(), replace=True)
            ])
        )

    def extract_mean_std(self, path: str, cls: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        transform = transforms.Compose([
            transforms.Resize((self.shape[-2], self.shape[-1])),
            transforms.ToTensor(),
        ])
        ds = ImageFolderDataset(
            path, 'unsupervised', self.raw_shape, self.ovr, self.nominal_label, self.anomalous_label,
            normal_classes=[cls], transform=transform, target_transform=transforms.Lambda(
                lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
            )
        )
        ds = Subset(
            ds,
            np.argwhere(
                np.isin(ds.targets, np.asarray([cls])) * np.isin(ds.anomaly_labels, np.asarray([self.nominal_label]))
            ).flatten().tolist()
        )
        loader = DataLoader(dataset=ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
        all_x = []
        for x, _ in loader:
            all_x.append(x)
        all_x = torch.cat(all_x)
        return all_x.permute(1, 0, 2, 3).flatten(1).mean(1), all_x.permute(1, 0, 2, 3).flatten(1).std(1)

    def check_data(self):
        # custom data check
        if not pt.exists(self.trainpath):
            raise ValueError(f'No custom data found since {self.trainpath} does not exist.')
        if not pt.exists(self.testpath):
            raise ValueError(f'No custom data found since {self.testpath} does not exist.')
        if self.ovr:
            if any([cls_dir.lower() in ('normal', 'nominal', 'anomalous') for cls_dir in os.listdir(self.trainpath)]):
                raise ValueError(
                    f'Found a class folder being named "normal", "nominal", or "anomalous" in ({self.trainpath}). '
                    f'Note that the class folders needs to match the class names (like "dog", "hazelnut"). '
                    f'Deactivate the one-vs-rest evaluation mode or change the class folders to class names.'
                )
        else:
            if any([cls_dir.lower() in ('normal', 'nominal', 'anomalous') for cls_dir in os.listdir(self.trainpath)]):
                raise ValueError(
                    f'Found a class folder being named "normal", "nominal", or "anomalous" in ({self.trainpath}). '
                    f'Note that the class folders needs to match the class names (like "dog", "hazelnut"). '
                    f'Normal samples need to be placed in CLASS_NAME/normal and anomalous samples in CLASS_NAME/anomalous. '
                )
            for split_dir in (self.trainpath, self.testpath):
                for cls_dir in os.listdir(split_dir):
                    if 'normal' not in [d.lower() for d in os.listdir(pt.join(split_dir, cls_dir))]:
                        raise ValueError(
                            f'All class folders need to contain a folder named "normal" for normal samples. '
                            f'However, did not find such a folder in {pt.join(split_dir, cls_dir)}.'
                        )
                    for lbl_dir in os.listdir(pt.join(split_dir, cls_dir)):
                        if lbl_dir.lower() not in ('normal', 'nominal', 'anomalous'):
                            raise ValueError(
                                f'All class folders need to contain folders for "normal" and "anomalous" data. '
                                f'However, found a folder named {lbl_dir} in {pt.join(split_dir, cls_dir)}.'
                            )
        train_classes = os.listdir(self.trainpath)
        test_classes = os.listdir(self.testpath)
        if train_classes != test_classes:
            raise ValueError(
                f'The training class names and test class names do no match. '
                f'The training class names are {train_classes} and the test class names {test_classes}.'
            )


class ImageFolderDataset(ImageFolder):
    def __init__(self, root: str, supervise_mode: str, raw_shape: Tuple[int, int, int], ovr: bool,
                 nominal_label: int, anomalous_label: int,
                 transform=None, target_transform=None, normal_classes=None, all_transform=None, ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if ovr:
            self.anomaly_labels = [self.target_transform(t) for t in self.targets]
        else:
            self.anomaly_labels = [
                nominal_label if f.split(os.sep)[-2].lower() in ['normal', 'nominal'] else anomalous_label
                for f, _ in self.samples
            ]
        self.normal_classes = normal_classes
        self.all_transform = all_transform  # contains the OnlineSupervisor
        self.supervise_mode = supervise_mode
        self.raw_shape = torch.Size(raw_shape)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        target = self.anomaly_labels[index]

        if self.target_transform is not None:
            pass  # already applied since we use self.anomaly_labels instead of self.targets

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
