import sys

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.online_superviser import OnlineSuperviser
from fcdd.datasets.preprocessing import MultiCompose
from torchvision.datasets import VOCDetection

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class ADPascalVoc(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0, preproc='ae',
                 supervise_mode='unsupervised', supervise_params=None, logger=None):
        super().__init__(root, logger=logger)
        assert normal_class == 0, 'One cls dataset with horse only!'
        if supervise_mode != 'unsupervised':
            assert supervise_params.get('online', True), \
                'ImageNet artificial anomaly generation needs to be applied online'
        supervise_params.update({'exclude_for_voc': MyPascalVoc.NAMES})

        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (3, 224, 224)
        self.raw_shape = (3, 224, 224)
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        assert supervise_params.get('nominal_label', 0) in [0, 1]
        self.nominal_label = supervise_params.get('nominal_label', 0)
        self.anomalous_label = 1 if self.nominal_label == 0 else 0
        self.normal_classes = tuple([self.nominal_label])

        if self.nominal_label != 0:
            print('Swapping labels, i.e. anomalies are 0 and nominals are 1.')

        # mean and std of original pictures
        mean = (0.4469, 0.4227, 0.3906)
        std = (0.2691, 0.2659, 0.2789)

        all_transform = []
        if preproc in ['', None, 'default', 'none']:
            test_transform = transform = transforms.Compose([
                transforms.Resize((self.shape[-1], self.raw_shape[-1])),  # not short edge because that skips watermark
                transforms.ToTensor(),
                transforms.Normalize(mean[normal_class], std[normal_class])
            ])
        elif preproc in ['aug1']:
            test_transform = transforms.Compose([
                transforms.Resize((self.raw_shape[-1], self.raw_shape[-1])),
                transforms.CenterCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform = transforms.Compose([
                transforms.Resize((self.raw_shape[-1], self.raw_shape[-1])),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                transforms.Normalize(mean, std)
            ])
        else:
            raise ValueError('Preprocessing set {} is not known.'.format(preproc))

        if supervise_params['online']:
            all_transform = MultiCompose([OnlineSuperviser(self, supervise_mode, supervise_params), *all_transform])
        else:
            all_transform = MultiCompose(all_transform)

        train_set = MyPascalVoc(root=self.root, split='train', download=True, nominal_label=self.nominal_label,
                                transform=transform, all_transform=all_transform, anomlbl=self.anomalous_label)

        self._generate_artificial_anomalies_train_set(
            supervise_mode if not supervise_params['online'] else 'unsupervised',
            supervise_params, train_set, normal_class
        )

        self._test_set = MyPascalVoc(root=self.root, split='val', download=True, nominal_label=self.nominal_label,
                                     transform=test_transform, anomlbl=self.anomalous_label)


class MyPascalVoc(VOCDetection):
    NAMES = [
        'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus',
        'car', 'motorbike', 'train', 'bottle', 'chair', 'table', 'plant', 'sofa', 'tv', 'monitor'
    ]

    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False, all_transform=None, nominal_label=0, anomlbl=1):
        super().__init__(root, '2007', split, download, transform, target_transform)
        self.all_transform = all_transform
        self.normal_classes = 0
        self.nominal_label = nominal_label
        self.anomalous_label = anomlbl
        self.targets = [
            self.parse_annotation(
                self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())['annotation']['object'], 'horse'
            ) for index in range(len(self.annotations))
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.targets[index]

        if self.target_transform is not None:  # online superviser assumes target transform to be already applied
            target = self.target_transform(target)

        # apply online superviser, if available
        if self.all_transform is not None:
            img, _, target = self.all_transform((transforms.ToTensor()(img), None, target))
            if isinstance(img, torch.Tensor):
                img = img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else img
                img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def parse_annotation(self, objects, normal_class_str):
        if isinstance(objects, dict):
            return self.nominal_label if objects['name'] == normal_class_str else self.anomalous_label
        else:  # list
            return self.nominal_label if any(obj['name'] == normal_class_str for obj in objects) \
                else self.anomalous_label
