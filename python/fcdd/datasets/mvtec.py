import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from fcdd.datasets.bases import TorchvisionDataset, GTSubset
from fcdd.datasets.mvtec_base import MvTec
from fcdd.datasets.online_supervisor import OnlineSupervisor
from fcdd.datasets.preprocessing import local_contrast_normalization, MultiCompose, get_target_label_idx
from fcdd.util.logging import Logger


class ADMvTec(TorchvisionDataset):
    enlarge = True  # enlarge dataset by repeating all data samples ten time, speeds up data loading

    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool,
                 logger: Logger = None, raw_shape: int = 240):
        """
        AD dataset for MVTec-AD. If no MVTec data is found in the root directory,
        the data is downloaded and processed to be stored in torch tensors with appropriate size (defined in raw_shape).
        This speeds up data loading at the start of training.
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
        :param raw_shape: the height and width of the raw MVTec images before passed through the preprocessing pipeline.
        """
        super().__init__(root, logger=logger)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (3, 224, 224)
        self.raw_shape = (3,) + (raw_shape, ) * 2
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 15))
        self.outlier_classes.remove(normal_class)
        assert nominal_label in [0, 1], 'GT maps are required to be binary!'
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        # min max after gcn l1 norm has been applied
        min_max_l1 = [
            [(-1.3336724042892456, -1.3107913732528687, -1.2445921897888184),
             (1.3779616355895996, 1.3779616355895996, 1.3779616355895996)],
            [(-2.2404820919036865, -2.3387579917907715, -2.2896201610565186),
             (4.573435306549072, 4.573435306549072, 4.573435306549072)],
            [(-3.184587001800537, -3.164201259613037, -3.1392977237701416),
             (1.6995097398757935, 1.6011602878570557, 1.5209171772003174)],
            [(-3.0334954261779785, -2.958242416381836, -2.7701096534729004),
             (6.503103256225586, 5.875098705291748, 5.814228057861328)],
            [(-3.100773334503174, -3.100773334503174, -3.100773334503174),
             (4.27892541885376, 4.27892541885376, 4.27892541885376)],
            [(-3.6565306186676025, -3.507692813873291, -2.7635035514831543),
             (18.966819763183594, 21.64590072631836, 26.408710479736328)],
            [(-1.5192601680755615, -2.2068002223968506, -2.3948357105255127),
             (11.564697265625, 10.976534843444824, 10.378695487976074)],
            [(-1.3207964897155762, -1.2889339923858643, -1.148416519165039),
             (6.854909896850586, 6.854909896850586, 6.854909896850586)],
            [(-0.9883341193199158, -0.9822461605072021, -0.9288841485977173),
             (2.290637969970703, 2.4007883071899414, 2.3044068813323975)],
            [(-7.236185073852539, -7.236185073852539, -7.236185073852539),
             (3.3777384757995605, 3.3777384757995605, 3.3777384757995605)],
            [(-3.2036616802215576, -3.221003532409668, -3.305514335632324),
             (7.022546768188477, 6.115569114685059, 6.310940742492676)],
            [(-0.8915618658065796, -0.8669204115867615, -0.8002046346664429),
             (4.4255571365356445, 4.642300128936768, 4.305730819702148)],
            [(-1.9086798429489136, -2.0004451274871826, -1.929288387298584),
             (5.463134765625, 5.463134765625, 5.463134765625)],
            [(-2.9547364711761475, -3.17536997795105, -3.143850803375244),
             (5.305514812469482, 4.535006523132324, 3.3618252277374268)],
            [(-1.2906527519226074, -1.2906527519226074, -1.2906527519226074),
             (2.515115737915039, 2.515115737915039, 2.515115737915039)]
        ]

        # mean and std of original images per class
        mean = [
            (0.53453129529953, 0.5307118892669678, 0.5491130352020264),
            (0.326835036277771, 0.41494372487068176, 0.46718254685401917),
            (0.6953922510147095, 0.6663950085639954, 0.6533040404319763),
            (0.36377236247062683, 0.35087138414382935, 0.35671544075012207),
            (0.4484519958496094, 0.4484519958496094, 0.4484519958496094),
            (0.2390524297952652, 0.17620408535003662, 0.17206747829914093),
            (0.3919542133808136, 0.2631213963031769, 0.22006843984127045),
            (0.21368788182735443, 0.23478130996227264, 0.24079132080078125),
            (0.30240726470947266, 0.3029524087905884, 0.32861486077308655),
            (0.7099748849868774, 0.7099748849868774, 0.7099748849868774),
            (0.4567880630493164, 0.4711957275867462, 0.4482630491256714),
            (0.19987481832504272, 0.18578395247459412, 0.19361256062984467),
            (0.38699793815612793, 0.276934415102005, 0.24219433963298798),
            (0.6718143820762634, 0.47696375846862793, 0.35050269961357117),
            (0.4014520049095154, 0.4014520049095154, 0.4014520049095154)
        ]
        std = [
            (0.3667600452899933, 0.3666728734970093, 0.34991779923439026),
            (0.15321789681911469, 0.21510766446590424, 0.23905669152736664),
            (0.23858436942100525, 0.2591284513473511, 0.2601949870586395),
            (0.14506031572818756, 0.13994529843330383, 0.1276693195104599),
            (0.1636597216129303, 0.1636597216129303, 0.1636597216129303),
            (0.1688646823167801, 0.07597383111715317, 0.04383210837841034),
            (0.06069392338395119, 0.04061736911535263, 0.0303945429623127),
            (0.1602524220943451, 0.18222476541996002, 0.15336430072784424),
            (0.30409011244773865, 0.30411985516548157, 0.28656429052352905),
            (0.1337062269449234, 0.1337062269449234, 0.1337062269449234),
            (0.12076705694198608, 0.13341768085956573, 0.12879984080791473),
            (0.22920562326908112, 0.21501320600509644, 0.19536510109901428),
            (0.20621345937252045, 0.14321941137313843, 0.11695228517055511),
            (0.08259467780590057, 0.06751163303852081, 0.04756828024983406),
            (0.32304847240448, 0.32304847240448, 0.32304847240448)
        ]

        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        img_gt_transform, img_gt_test_transform = None, None
        all_transform = []
        if preproc == 'lcn':
            assert self.raw_shape == self.shape, 'in case of no augmentation, raw shape needs to fit net input shape'
            img_gt_transform = img_gt_test_transform = MultiCompose([
                transforms.ToTensor(),
            ])
            test_transform = transform = transforms.Compose([
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    min_max_l1[normal_class][0],
                    [ma - mi for ma, mi in zip(min_max_l1[normal_class][1], min_max_l1[normal_class][0])]
                )
            ])
        elif preproc in ['', None, 'default', 'none']:
            assert self.raw_shape == self.shape, 'in case of no augmentation, raw shape needs to fit net input shape'
            img_gt_transform = img_gt_test_transform = MultiCompose([
                transforms.ToTensor(),
            ])
            test_transform = transform = transforms.Compose([
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['aug1']:
            img_gt_transform = MultiCompose([
                transforms.RandomChoice(
                    [transforms.RandomCrop(self.shape[-1], padding=0), transforms.Resize(self.shape[-1], Image.NEAREST)]
                ),
                transforms.ToTensor(),
            ])
            img_gt_test_transform = MultiCompose(
                [transforms.Resize(self.shape[-1], Image.NEAREST), transforms.ToTensor()]
            )
            test_transform = transforms.Compose([
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomChoice([
                    transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                    transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
                ]),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: (x + torch.randn_like(x).mul(np.random.randint(0, 2)).mul(x.std()).mul(0.1)).clamp(0, 1)
                ),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['lcnaug1']:
            img_gt_transform = MultiCompose([
                transforms.RandomChoice(
                    [transforms.RandomCrop(self.shape[-1], padding=0), transforms.Resize(self.shape[-1], Image.NEAREST)]
                ),
                transforms.ToTensor(),
            ])
            img_gt_test_transform = MultiCompose(
                [transforms.Resize(self.shape[-1], Image.NEAREST), transforms.ToTensor()]
            )
            test_transform = transforms.Compose([
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    min_max_l1[normal_class][0],
                    [ma - mi for ma, mi in zip(min_max_l1[normal_class][1], min_max_l1[normal_class][0])]
                )
            ])
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomChoice([
                        transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                        transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
                ]),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: (x + torch.randn_like(x).mul(np.random.randint(0, 2)).mul(x.std()).mul(0.1)).clamp(0, 1)
                ),
                transforms.Lambda(lambda x: local_contrast_normalization(x, scale='l1')),
                transforms.Normalize(
                    min_max_l1[normal_class][0],
                    [ma - mi for ma, mi in zip(min_max_l1[normal_class][1], min_max_l1[normal_class][0])]
                )
            ])
        else:
            raise ValueError('Preprocessing pipeline {} is not known.'.format(preproc))

        target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        )

        if online_supervision:
            # order: target_transform -> all_transform -> img_gt transform -> transform
            assert supervise_mode not in ['supervised'], 'supervised mode works only offline'
            all_transform = MultiCompose([
                *all_transform,
                OnlineSupervisor(self, supervise_mode, noise_mode, oe_limit),
            ])
            train_set = MvTec(
                root=self.root, split='train', download=True,
                target_transform=target_transform,
                img_gt_transform=img_gt_transform, transform=transform, all_transform=all_transform,
                shape=self.raw_shape, normal_classes=self.normal_classes,
                nominal_label=self.nominal_label, anomalous_label=self.anomalous_label,
                enlarge=ADMvTec.enlarge
            )
            self._train_set = GTSubset(
                train_set, get_target_label_idx(train_set.targets.clone().data.cpu().numpy(), self.normal_classes)
            )
            test_set = MvTec(
                root=self.root, split='test_anomaly_label_target', download=True,
                target_transform=transforms.Lambda(
                    lambda x: self.anomalous_label if x != MvTec.normal_anomaly_label_idx else self.nominal_label
                ),
                img_gt_transform=img_gt_test_transform, transform=test_transform, shape=self.raw_shape,
                normal_classes=self.normal_classes,
                nominal_label=self.nominal_label, anomalous_label=self.anomalous_label,
                enlarge=False
            )
            test_idx_normal = get_target_label_idx(test_set.targets.clone().data.cpu().numpy(), self.normal_classes)
            self._test_set = GTSubset(test_set, test_idx_normal)
        else:
            all_transform = MultiCompose([
                *all_transform,
            ]) if len(all_transform) > 0 else None
            train_set = MvTec(
                root=self.root, split='train', download=True,
                target_transform=target_transform, all_transform=all_transform,
                img_gt_transform=img_gt_transform, transform=transform, shape=self.raw_shape,
                normal_classes=self.normal_classes,
                nominal_label=self.nominal_label, anomalous_label=self.anomalous_label,
                enlarge=ADMvTec.enlarge
            )
            test_set = MvTec(
                root=self.root, split='test_anomaly_label_target', download=True,
                target_transform=transforms.Lambda(
                    lambda x: self.anomalous_label if x != MvTec.normal_anomaly_label_idx else self.nominal_label
                ),
                img_gt_transform=img_gt_test_transform, transform=test_transform, shape=self.raw_shape,
                normal_classes=self.normal_classes,
                nominal_label=self.nominal_label, anomalous_label=self.anomalous_label, enlarge=False
            )
            test_idx_normal = get_target_label_idx(test_set.targets.clone().data.cpu().numpy(), self.normal_classes)
            self._test_set = GTSubset(test_set, test_idx_normal)
            self._generate_artificial_anomalies_train_set(supervise_mode, noise_mode, oe_limit, train_set, normal_class)

