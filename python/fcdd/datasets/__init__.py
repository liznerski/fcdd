from copy import deepcopy
from typing import List
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.cifar import ADCIFAR10
from fcdd.datasets.fmnist import ADFMNIST
from fcdd.datasets.imagenet import ADImageNet
from fcdd.datasets.mvtec import ADMvTec
from fcdd.datasets.pascal_voc import ADPascalVoc
from fcdd.datasets.image_folder import ADImageFolderDataset
from fcdd.datasets.image_folder_gtms import ADImageFolderDatasetGTM

DS_CHOICES = ('mnist', 'cifar10', 'fmnist', 'mvtec', 'imagenet', 'pascalvoc', 'custom')
PREPROC_CHOICES = (
    'lcn', 'lcnaug1', 'aug1', 'aug1_blackcenter', 'aug1_blackcenter_inverted', 'none'
)
CUSTOM_CLASSES = []


def load_dataset(dataset_name: str, data_path: str, normal_class: int, preproc: str,
                 supervise_mode: str, noise_mode: str, online_supervision: bool, nominal_label: int,
                 oe_limit: int, logger=None) -> TorchvisionDataset:
    """ Loads the dataset with given preprocessing pipeline and supervise parameters """

    assert dataset_name in DS_CHOICES
    assert preproc in PREPROC_CHOICES

    dataset = None

    if dataset_name == 'cifar10':
        dataset = ADCIFAR10(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
            oe_limit=oe_limit, logger=logger, nominal_label=nominal_label
        )
    elif dataset_name == 'fmnist':
        dataset = ADFMNIST(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
            oe_limit=oe_limit, logger=logger, nominal_label=nominal_label
        )
    elif dataset_name == 'mvtec':
        dataset = ADMvTec(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
            oe_limit=oe_limit, logger=logger, nominal_label=nominal_label
        )
    elif dataset_name == 'imagenet':
        dataset = ADImageNet(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
            oe_limit=oe_limit, logger=logger, nominal_label=nominal_label
        )
    elif dataset_name == 'pascalvoc':
        dataset = ADPascalVoc(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
            oe_limit=oe_limit, logger=logger, nominal_label=nominal_label
        )
    elif dataset_name == 'custom':
        if ADImageFolderDataset.gtm:
            dataset = ADImageFolderDatasetGTM(
                root=data_path, normal_class=normal_class, preproc=preproc,
                supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
                oe_limit=oe_limit, logger=logger, nominal_label=nominal_label
            )
        else:
            dataset = ADImageFolderDataset(
                root=data_path, normal_class=normal_class, preproc=preproc,
                supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
                oe_limit=oe_limit, logger=logger, nominal_label=nominal_label
            )
    else:
        raise NotImplementedError(f'Dataset {dataset_name} is unknown.')

    return dataset


def no_classes(dataset_name: str) -> int:
    return {
        'cifar10': 10,
        'fmnist': 10,
        'mvtec': 15,
        'imagenet': 30,
        'pascalvoc': 1,
        'custom': len(CUSTOM_CLASSES)
    }[dataset_name]


def str_labels(dataset_name: str) -> List[str]:
    return {
        'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'fmnist': [
            't-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ],
        'mvtec': [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
            'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
            'wood', 'zipper'
        ],
        'imagenet': deepcopy(ADImageNet.ad_classes),
        'pascalvoc': ['horse'],
        'custom': list(CUSTOM_CLASSES)
    }[dataset_name]
