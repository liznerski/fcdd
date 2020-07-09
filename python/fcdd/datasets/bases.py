from abc import ABC, abstractmethod

import numpy as np
import torch
from fcdd.datasets.noise_modes import generate_noise
from fcdd.datasets.offline_superviser import noise as apply_noise, malformed_normal as apply_malformed_normal
from fcdd.datasets.preprocessing import get_target_label_idx
from torch.utils.data import DataLoader
from torch.utils.data import Subset


class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str, logger=None):
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
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__

    def logprint(self, s, fps=False):
        if self.logger is not None:
            self.logger.print(s, fps)
        else:
            print(s)


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    def get_train_set(self, classes=None):
        if classes is not None and len(classes) > 0:
            labels = self.train_set.targets.clone().data.cpu().numpy()
            idx = np.argwhere(np.isin(labels, classes)).flatten().tolist()
            ret = Subset(self.train_set, idx)
        else:
            ret = self.train_set
        return ret

    def __init__(self, root: str, logger=None):
        super().__init__(root, logger=logger)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0, classes=None) -> (
            DataLoader, DataLoader):
        assert not shuffle_test, \
            'using shuffled test raises problems with original GT maps for GT datasets, thus disabled atm!'
        # classes = None means all classes
        train_loader = DataLoader(dataset=self.get_train_set(classes), batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, pin_memory=True,)
        return train_loader, test_loader

    def preview(self, percls=20, train=True):
        self.logprint('Generating dataset preview...')
        if train:
            loader, _ = self.loaders(20, num_workers=4)
        else:
            _, loader = self.loaders(20, num_workers=4)
        all_x, all_y, all_gts, out = [], [], [], []
        if isinstance(self.train_set, GTMapADDataset):
            for x, y, gts in loader:
                all_x.append(x), all_y.append(y), all_gts.append(gts)
        else:
            for x, y in loader:
                all_x.append(x), all_y.append(y)
        x, y, gts = torch.cat(all_x), torch.cat(all_y), torch.cat(all_gts) if len(all_gts) > 0 else None
        for c in sorted(set(y.tolist())):
            out.append(x[y == c][:percls])
        if gts is not None:
            assert len(set(gts.reshape(-1).tolist())) <= 2, 'training process assumes zero-one gtmaps'
            out.append(torch.zeros_like(x[:percls]))
            for c in sorted(set(y.tolist())):
                g = gts[y == c][:percls]
                if x.shape[1] > 1:
                    g = g.repeat(1, x.shape[1], 1, 1)
                out.append(g)
        self.logprint('Dataset preview generated.')
        return torch.cat(out)

    def _generate_artificial_anomalies_train_set(self, supervise_mode, supervise_params, train_set, nom_class):
        """
        Method to generate offline anomalies, i.e. generate them once at the start of the training and add
        it to the train set. This is way faster, but lacks diversity.
        :param supervise_mode: generate anomalies based on mode,
            unsupervised: no anomalies
            other: other classes, i.e. all the true anomalies!
            noise: pure noise images
            malformed_normal: add noise to nominal samples
            malformed_normal_gt: add noise to nominal samples and store positions in an artificial ground-truth map
        :param supervise_params:
        :param train_set:
        :param nom_class:
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
                supervise_params.get('noise_mode', None), train_set.data[train_idx_normal].shape, supervise_params,
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
            self._train_set = apply_noise(self, generated_noise, norm, nom_class, train_set)
        elif supervise_mode in ['malformed_normal']:
            self._train_set = apply_malformed_normal(self, generated_noise, norm, nom_class, train_set)
        elif supervise_mode in ['malformed_normal_gt']:
            train_set, gtmaps = apply_malformed_normal(self, generated_noise, norm, nom_class, train_set, gt=True)
            self._train_set = GTMapADDatasetExtension(train_set, gtmaps)
        else:
            raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
        if supervise_mode not in ['unsupervised', 'other']:
            self.logprint('Artificial anomalies generated.')

    def _generate_noise(self, noise_mode, size, params=None, datadir=None):
        generated_noise = generate_noise(noise_mode, size, params, logger=self.logger, datadir=datadir)
        return generated_noise


class ThreeReturnsDataset(object):
    @abstractmethod
    def __getitem__(self, index):
        return None, None, None


class GTMapADDataset(ThreeReturnsDataset):
    @abstractmethod
    def __getitem__(self, index):
        x, y, gtmap = None, None, None
        return x, y, gtmap


class GTSubset(Subset, GTMapADDataset):
    pass


class GTMapADDatasetExtension(GTMapADDataset):
    """
    Given a dataset, uses the dataset to return tuples per its __getitem__, but adds a last item gtmaps[idx] to it
    :param overwrite:
        If dataset is already a GTMapADDataset itself, determines if gtmaps of dataset shall be overwritten.
        None values of found gtmaps in dataset are overwritten in any case.
    """
    def __init__(self, dataset, gtmaps, overwrite=True):
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

    def __getitem__(self, index):
        """
        Adds a third return, the ground truth map, to the return of standard mnist dataset.
        """
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
