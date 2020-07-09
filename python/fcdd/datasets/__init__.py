from fcdd.datasets.cifar import ADCIFAR10
from fcdd.datasets.fmnist import ADFMNIST
from fcdd.datasets.mvtec import ADMvTec
from fcdd.datasets.imagenet import ADImageNet
from fcdd.datasets.pascal_voc import ADPascalVoc
from copy import deepcopy

DS_CHOICES = ('mnist', 'cifar10', 'fmnist', 'mvtec', 'imagenet', 'pascalvoc')
PREPROC_CHOICES = (
    'ae', 'default', 'aug1', 'aug2', 'aeaug1', 'aeauganomonly', 'aeaug1_blackcenter', 'aeaug1_blackcenter_inverted',
    'aug1_blackcenter', 'aug1_blackcenter_inverted'
)


def load_dataset(dataset_name, data_path, normal_class, preproc='ae',
                 supervise_mode='unsupervised', supervise_params=None, raw_shape=240, logger=None):
    """Loads the dataset."""

    assert dataset_name in DS_CHOICES
    assert preproc in PREPROC_CHOICES

    dataset = None

    if dataset_name == 'cifar10':
        dataset = ADCIFAR10(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, supervise_params=supervise_params, logger=logger
        )
    elif dataset_name == 'fmnist':
        dataset = ADFMNIST(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, supervise_params=supervise_params, logger=logger
        )
    elif dataset_name == 'mvtec':
        dataset = ADMvTec(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, supervise_params=supervise_params,
            raw_shape=raw_shape, logger=logger
        )
    elif dataset_name == 'imagenet':
        dataset = ADImageNet(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, supervise_params=supervise_params, logger=logger
        )
    elif dataset_name == 'pascalvoc':
        dataset = ADPascalVoc(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, supervise_params=supervise_params, logger=logger
        )

    return dataset


def no_classes(dataset_name):
    return {
        'cifar10': 10,
        'fmnist': 10,
        'mvtec': 15,
        'imagenet': 30,
        'pascalvoc': 1,
    }[dataset_name]


def str_labels(dataset_name):
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
        'pascalvoc': ['horse']
    }[dataset_name]
