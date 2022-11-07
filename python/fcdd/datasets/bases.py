from abc import ABC, abstractmethod
from typing import Tuple
from collections import Counter

import numpy as np
import torch
from fcdd.datasets.noise_modes import generate_noise
from fcdd.datasets.offline_supervisor import noise as apply_noise, malformed_normal as apply_malformed_normal
from fcdd.datasets.preprocessing import get_target_label_idx
from fcdd.util.logging import Logger
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset


class BaseADDataset(ABC):
    """ Anomaly detection dataset base class """

    def __init__(self, root: str, logger: Logger = None):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self._train_set = None  # must be of type torch.utils.data.Dataset
        self._test_set = None  # must be of type torch.utils.data.Dataset

        self.shape = None  # shape of datapoints, c x h x w
        self.raw_shape = None  # shape of datapoint before preprocessing is applied, c x h x w

        self.logger = logger

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> Tuple[
            DataLoader, DataLoader]:
        """ Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set. """
        pass

    def __repr__(self):
        return self.__class__.__name__

    def logprint(self, s: str, fps: bool = False):
        """ prints a string via the logger """
        if self.logger is not None:
            self.logger.print(s, fps)
        else:
            print(s)


class TorchvisionDataset(BaseADDataset):
    """ TorchvisionDataset class for datasets already implemented in torchvision.datasets """

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    def __init__(self, root: str, logger=None):
        super().__init__(root, logger=logger)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0)\
            -> Tuple[DataLoader, DataLoader]:
        assert not shuffle_test, \
            'using shuffled test raises problems with original GT maps for GT datasets, thus disabled atm!'
        # classes = None means all classes
        # TODO: persistent_workers=True makes training significantly faster,
        #  but rn this sometimes yields "OSError: [Errno 12] Cannot allocate memory" as there seems to be a memory leak
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, pin_memory=False, persistent_workers=False)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, pin_memory=False, persistent_workers=False)
        return train_loader, test_loader

    def preview(self, percls=20, train=True, classes=(0, 1)) -> torch.Tensor:
        """
        Generates a preview of the dataset, i.e. it generates an image of some randomly chosen outputs
        of the dataloader, including ground-truth maps if available.
        The data samples already have been augmented by the preprocessing pipeline.
        This method is useful to have an overview of how the preprocessed samples look like and especially
        to have an early look at the artificial anomalies.
        :param percls: how many samples are shown per class, i.e. for anomalies and nominal samples each
        :param train: whether to show training samples or test samples
        :param classes: the class labels for which images are collected. Defaults to (0, 1) for normal and anomalous.
        :return: a Tensor of images (n x c x h x w)
        """
        self.logprint('Generating dataset preview...')
        # assert num_workers>0, otherwise the OnlineSupervisor is initialized with the same shuffling in later workers
        if train:
            loader, _ = self.loaders(10, num_workers=1, shuffle_train=True)
        else:
            _, loader = self.loaders(10, num_workers=1, shuffle_test=True)
        x, y, gts, out = torch.FloatTensor(), torch.LongTensor(), torch.FloatTensor(), []
        if isinstance(self.train_set, GTMapADDataset):
            for xb, yb, gtsb in loader:
                x, y, gts = torch.cat([x, xb]), torch.cat([y, yb]), torch.cat([gts, gtsb])
                if all([x[y == c].size(0) >= percls for c in classes]):
                    break
        else:
            for xb, yb in loader:
                x, y = torch.cat([x, xb]), torch.cat([y, yb])
                if all([x[y == c].size(0) >= percls for c in classes]):
                    break
        for c in sorted(set(y.tolist())):
            out.append(x[y == c][:percls])
        if len(gts) > 0:
            assert len(set(gts.reshape(-1).tolist())) <= 2, 'training process assumes zero-one gtmaps'
            out.append(torch.zeros_like(x[y == 0][:percls]))
            for c in sorted(set(y.tolist())):
                g = gts[y == c][:percls]
                if x.shape[1] > 1:
                    g = g.repeat(1, x.shape[1], 1, 1)
                out.append(g)
        self.logprint('Dataset preview generated.')
        return torch.stack([o[:min(Counter(y.tolist()).values())] for o in out])

    def _generate_artificial_anomalies_train_set(self, supervise_mode: str, noise_mode: str, oe_limit: int,
                                                 train_set: Dataset, nom_class: int):
        """
        This method generates offline artificial anomalies,
        i.e. it generates them once at the start of the training and adds them to the training set.
        It creates a balanced dataset, thus sampling as many anomalies as there are nominal samples.
        This is way faster than online generation, but lacks diversity (hence usually weaker performance).
        :param supervise_mode: the type of generated artificial anomalies.
            unsupervised: no anomalies, returns a subset of the original dataset containing only nominal samples.
            other: other classes, i.e. all the true anomalies!
            noise: pure noise images (can also be outlier exposure based).
            malformed_normal: add noise to nominal samples to create malformed nominal anomalies.
            malformed_normal_gt: like malformed_normal, but also creates artificial ground-truth maps
                that mark pixels anomalous where the difference between the original nominal sample
                and the malformed one is greater than a low threshold.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: the number of different outlier exposure samples used in case of outlier exposure based noise.
        :param train_set: the training set that is to be extended with artificial anomalies.
        :param nom_class: the class considered nominal
        :return:
        """
        if isinstance(train_set.targets, torch.Tensor):
            dataset_targets = train_set.targets.clone().data.cpu().numpy()
        else:  # e.g. imagenet
            dataset_targets = np.asarray(train_set.targets)
        train_idx_normal = get_target_label_idx(dataset_targets, self.normal_classes)
        generated_noise = norm = None
        if supervise_mode not in ['unsupervised', 'other']:
            self.logprint('Generating artificial anomalies...')
            generated_noise = self._generate_noise(
                noise_mode, train_set.data[train_idx_normal].shape, oe_limit,
                self.root
            )
            norm = train_set.data[train_idx_normal]
        if supervise_mode in ['other']:
            self._train_set = train_set
        elif supervise_mode in ['unsupervised']:
            if isinstance(train_set, GTMapADDataset):
                self._train_set = GTSubset(train_set, train_idx_normal)
            else:
                self._train_set = Subset(train_set, train_idx_normal)
        elif supervise_mode in ['noise']:
            self._train_set = apply_noise(self.outlier_classes, generated_noise, norm, nom_class, train_set)
        elif supervise_mode in ['malformed_normal']:
            self._train_set = apply_malformed_normal(self.outlier_classes, generated_noise, norm, nom_class, train_set)
        elif supervise_mode in ['malformed_normal_gt']:
            train_set, gtmaps = apply_malformed_normal(
                self.outlier_classes, generated_noise, norm, nom_class, train_set, gt=True
            )
            self._train_set = GTMapADDatasetExtension(train_set, gtmaps)
        else:
            raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
        if supervise_mode not in ['unsupervised', 'other']:
            self.logprint('Artificial anomalies generated.')

    def _generate_noise(self, noise_mode: str, size: torch.Size, oe_limit: int = None, datadir: str = None) -> torch.Tensor:
        generated_noise = generate_noise(noise_mode, size, oe_limit, logger=self.logger, datadir=datadir)
        return generated_noise


class ThreeReturnsDataset(Dataset):
    """ Dataset base class returning a tuple of three items as data samples """
    @abstractmethod
    def __getitem__(self, index):
        return None, None, None


class GTMapADDataset(ThreeReturnsDataset):
    """ Dataset base class returning a tuple (input, label, ground-truth map) as data samples """
    @abstractmethod
    def __getitem__(self, index):
        x, y, gtmap = None, None, None
        return x, y, gtmap


class GTSubset(Subset, GTMapADDataset):
    """ Subset base class for GTMapADDatasets """
    pass


class GTMapADDatasetExtension(GTMapADDataset):
    """
    This class is used to extend a regular torch dataset such that is returns the corresponding ground-truth map
    in addition to the usual (input, label) tuple.
    """
    def __init__(self, dataset: Dataset, gtmaps: torch.Tensor, overwrite=True):
        """
        :param dataset: a regular torch dataset
        :param gtmaps: a tensor of ground-truth maps (n x h x w)
        :param overwrite: if dataset is already a GTMapADDataset itself,
            determines if gtmaps of dataset shall be overwritten.
            None values of found gtmaps in dataset are overwritten in any case.
        """
        self.ds = dataset
        self.extended_gtmaps = gtmaps
        self.overwrite = overwrite
        if isinstance(self.ds, GTMapADDataset):
            assert hasattr(self.ds, 'gt')
            if self.ds.gt is None:
                self.ds.gt = gtmaps.mul(255).byte()
                self.overwrite = False

    @property
    def targets(self):
        return self.ds.targets

    @property
    def data(self):
        return self.ds.data

    def __getitem__(self, index: int):
        gtmap = self.extended_gtmaps[index]

        if isinstance(self.ds, GTMapADDataset):
            x, y, gt = self.ds[index]
            if self.overwrite or gt is None:
                gt = gtmap
            res = (x, y, gt)
        else:
            res = (*self.ds[index], gtmap)

        return res

    def __len__(self):
        return len(self.ds)
