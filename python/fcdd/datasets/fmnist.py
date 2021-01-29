from typing import Tuple

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.online_supervisor import OnlineSupervisor
from fcdd.datasets.preprocessing import local_contrast_normalization, MultiCompose
from fcdd.util.logging import Logger
from torchvision.datasets import FashionMNIST


class ADFMNIST(TorchvisionDataset):
    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool, logger: Logger = None):
        """
        AD dataset for Fashion-MNIST.
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
        super().__init__(root, logger=logger)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (1, 28, 28)
        self.raw_shape = (28, 28)
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        # Pre-computed min and max values (after applying LCN) from train data per class
        min_max_l1 = [
            (-2.681239128112793, 24.85430908203125),
            (-2.5778584480285645, 11.169795989990234),
            (-2.808171510696411, 19.133548736572266),
            (-1.9533653259277344, 18.65673065185547),
            (-2.6103856563568115, 19.166685104370117),
            (-1.2358522415161133, 28.463092803955078),
            (-3.251605987548828, 24.196823120117188),
            (-1.0814440250396729, 16.878812789916992),
            (-3.6560964584350586, 11.3502836227417),
            (-1.3859291076660156, 11.426650047302246)
        ]

        # mean and std of original images per class
        mean = [
            [0.3256056010723114],
            [0.22290456295013428],
            [0.376699835062027],
            [0.25889596343040466],
            [0.3853232264518738],
            [0.1367349475622177],
            [0.3317836821079254],
            [0.16769391298294067],
            [0.35355499386787415],
            [0.30119451880455017]
        ]
        std = [
            [0.35073918104171753],
            [0.34353047609329224],
            [0.3586803078651428],
            [0.3542196452617645],
            [0.37631189823150635],
            [0.26310813426971436],
            [0.3392786681652069],
            [0.29478660225868225],
            [0.3652712106704712],
            [0.37053292989730835]
        ]

        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        if preproc == 'lcn':
            test_transform = transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    [min_max_l1[normal_class][0]], [min_max_l1[normal_class][1] - min_max_l1[normal_class][0]]
                )
            ])
        elif preproc in ['', None, 'default', 'none']:
            test_transform = transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['aug1']:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomCrop(28, padding=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['lcnaug1']:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    [min_max_l1[normal_class][0]], [min_max_l1[normal_class][1] - min_max_l1[normal_class][0]]
                )
            ])
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    [min_max_l1[normal_class][0]], [min_max_l1[normal_class][1] - min_max_l1[normal_class][0]]
                )
            ])
        else:
            raise ValueError('Preprocessing pipeline {} is not known.'.format(preproc))

        target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )
        all_transform = None
        if online_supervision:
            if noise_mode not in ['emnist']:
                self.raw_shape = (1, 28, 28)
                all_transform = MultiCompose([
                    OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit),
                    transforms.Lambda(lambda x: x.squeeze() if isinstance(x, torch.Tensor) else x)
                ])
            else:
                all_transform = MultiCompose([OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit), ])
            self.raw_shape = (28, 28)

        train_set = MyFashionMNIST(root=self.root, train=True, download=True, normal_classes=self.normal_classes,
                                   transform=transform, target_transform=target_transform, all_transform=all_transform)

        self._generate_artificial_anomalies_train_set(
            supervise_mode if not online_supervision else 'unsupervised',  # gets rid of true anomalous samples
            noise_mode, oe_limit, train_set, normal_class
        )

        self._test_set = MyFashionMNIST(root=self.root, train=False, download=True, normal_classes=self.normal_classes,
                                        transform=test_transform, target_transform=target_transform)


class MyFashionMNIST(FashionMNIST):
    """ Fashion-MNIST dataset extension, s.t. target_transform and online supervisor is applied """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, normal_classes=None, all_transform=None):
        super().__init__(root, train, transform, target_transform, download)
        self.all_transform = all_transform
        self.normal_classes = normal_classes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[index], int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        # apply online supervisor, if available
        if self.all_transform is not None:
            img, _, target = self.all_transform((img, None, target))
            img = img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else img

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
