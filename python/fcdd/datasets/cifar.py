from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.online_supervisor import OnlineSupervisor
from fcdd.datasets.preprocessing import local_contrast_normalization, MultiCompose, BlackCenter
from fcdd.util.logging import Logger
from torchvision.datasets import CIFAR10


class ADCIFAR10(TorchvisionDataset):
    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool, logger: Logger = None):
        """
        AD dataset for Cifar-10.
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
            or offline before training (same for all epochs in this case).
        :param logger: logger
        """
        super().__init__(root, logger=logger)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (3, 32, 32)
        self.raw_shape = (3, 32, 32)
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        if self.nominal_label != 0:
            print('Swapping labels, i.e. anomalies are 0 and nominals are 1.')

        # Pre-computed min and max values (after applying LCN) from train data per class
        min_max_l1 = [
            (-28.94083453598571, 13.802961825439636),
            (-6.681770233365245, 9.158067708230273),
            (-34.924463588638204, 14.419298165027628),
            (-10.599172931391799, 11.093187820377565),
            (-11.945022995801637, 10.628045447867583),
            (-9.691969487694928, 8.948326776180823),
            (-9.174940012342555, 13.847014686472365),
            (-6.876682005899029, 12.282371383343161),
            (-15.603507135507172, 15.2464923804279),
            (-6.132882973622672, 8.046098172351265)
        ]

        # mean and std of original images per class
        mean = [
            [0.5256516933441162, 0.5603281855583191, 0.5888723731040955],
            [0.4711322784423828, 0.45446228981018066, 0.4471212327480316],
            [0.48923906683921814, 0.49146366119384766, 0.423904687166214],
            [0.4954785108566284, 0.45636114478111267, 0.4154069721698761],
            [0.47155335545539856, 0.46515223383903503, 0.37797248363494873],
            [0.49992093443870544, 0.4646056592464447, 0.4164286255836487],
            [0.47001829743385315, 0.43829214572906494, 0.34500396251678467],
            [0.5019531846046448, 0.47983652353286743, 0.4167139232158661],
            [0.4902143180370331, 0.5253947973251343, 0.5546804070472717],
            [0.4986417591571808, 0.4852965474128723, 0.4780091941356659]
        ]
        std = [
            [0.2502202093601227, 0.24083486199378967, 0.2659735083580017],
            [0.26806357502937317, 0.2658274173736572, 0.2749459445476532],
            [0.22705480456352234, 0.2209445983171463, 0.24337927997112274],
            [0.2568431496620178, 0.25227081775665283, 0.25799375772476196],
            [0.21732737123966217, 0.20652702450752258, 0.21182335913181305],
            [0.2504253387451172, 0.24374878406524658, 0.2489463835954666],
            [0.22888341546058655, 0.21856172382831573, 0.2204199582338333],
            [0.2430490106344223, 0.243973046541214, 0.25171563029289246],
            [0.24962472915649414, 0.24068884551525116, 0.25149762630462646],
            [0.2680525481700897, 0.26910799741744995, 0.2810165584087372]
        ]

        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        # also contains options for the black center experiments
        all_transform = []
        if preproc == 'lcn':
            test_transform = transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    [min_max_l1[normal_class][0]] * 3, [min_max_l1[normal_class][1] - min_max_l1[normal_class][0]] * 3
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
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['aug1_blackcenter']:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                BlackCenter(0.6),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['aug1_blackcenter_inverted']:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                BlackCenter(0.6, inverse=True),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        else:
            raise ValueError('Preprocessing pipeline {} is not known.'.format(preproc))

        target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )
        if online_supervision:
            all_transform = MultiCompose([OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit), *all_transform])
        else:
            all_transform = MultiCompose(all_transform)

        train_set = MYCIFAR10(root=self.root, train=True, download=True, normal_classes=self.normal_classes,
                              transform=transform, target_transform=target_transform, all_transform=all_transform)
        train_set.targets = torch.from_numpy(np.asarray(train_set.targets))
        train_set.data = torch.from_numpy(train_set.data).transpose(1, 3).transpose(2, 3)

        self._generate_artificial_anomalies_train_set(
            supervise_mode if not online_supervision else 'unsupervised', noise_mode,
            oe_limit, train_set, normal_class
        )

        self._test_set = MYCIFAR10(root=self.root, train=False, download=True, normal_classes=self.normal_classes,
                                   transform=test_transform, target_transform=target_transform)


class MYCIFAR10(CIFAR10):
    """ Cifar-10 dataset extension, s.t. target_transform and online supervisor is applied """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, all_transform=None, normal_classes=None):
        super().__init__(root, train, transform, target_transform, download)
        self.all_transform = all_transform
        self.normal_classes = normal_classes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.target_transform is not None:  # online supervisor assumes target transform to be already applied
            target = self.target_transform(target)

        # apply online supervisor, if available
        if self.all_transform is not None:
            img, _, target = self.all_transform((img, None, target))
            img = img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else img

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


