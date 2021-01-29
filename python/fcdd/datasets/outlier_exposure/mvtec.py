import numpy as np
import torch
import torchvision.transforms as transforms
from fcdd.datasets.mvtec_base import MvTec
from fcdd.datasets.preprocessing import MultiCompose, get_target_label_idx
from fcdd.util.logging import Logger
from torch.utils.data import DataLoader
from typing import List, Tuple, Union


def ceil(x: float):
    return int(np.ceil(x))


class OEMvTec(MvTec):
    def __init__(self, size: torch.Size, clsses: List[int], root: str = None, limit_var: int = np.infty,
                 limit_per_anomaly=True, download=True, logger: Logger = None, gt=False, remove_nominal=True):
        """
        Outlier Exposure dataset for MVTec-AD. Considers only a part of the classes.
        :param size: size of the samples in n x c x h x w, samples will be resized to h x w. If n is larger than the
            number of samples available in MVTec-AD, dataset will be enlarged by repetitions to fit n.
            This is important as exactly n images are extracted per iteration of the data_loader.
            For online supervision n should be set to 1 because only one sample is extracted at a time.
        :param clsses: the classes that are to be considered, i.e. all other classes are dismissed.
        :param root: root directory where data is found or is to be downloaded to.
        :param limit_var: limits the number of different samples, i.e. randomly chooses limit_var many samples
            from all available ones to be the training data.
        :param limit_per_anomaly: whether limit_var limits the number of different samples per type
            of defection or overall.
        :param download: whether to download the data if it is not found in root.
        :param logger: logger.
        :param gt: whether ground-truth maps are to be included in the data.
        :param remove_nominal: whether nominal samples are to be excluded from the data.
        """
        assert len(size) == 4 and size[2] == size[3]
        assert size[1] in [1, 3]
        self.root = root
        self.logger = logger
        self.size = size
        self.use_gt = gt
        self.clsses = clsses
        super().__init__(root, 'test', download=download, shape=size[1:], logger=logger)

        self.img_gt_transform = MultiCompose([
            transforms.Resize((size[2], size[2])),
            transforms.ToTensor()
        ])
        self.picks = get_target_label_idx(self.targets, self.clsses)
        if remove_nominal:
            self.picks = sorted(list(set.intersection(
                set(self.picks),
                set((self.anomaly_labels != self.normal_anomaly_label_idx).nonzero().squeeze().tolist())
            )))
        if limit_per_anomaly and limit_var is not None:
            new_picks = []
            for l in set(self.anomaly_labels.tolist()):
                linclsses = list(set.intersection(
                    set(self.picks), set((self.anomaly_labels == l).nonzero().squeeze().tolist())
                ))
                if len(linclsses) == 0:
                    continue
                if limit_var < len(linclsses):
                    new_picks.extend(np.random.choice(linclsses, size=limit_var, replace=False))
                else:
                    self.logprint(
                        'OEMvTec shall be limited to {} samples per anomaly label, '
                        'but MvTec anomaly label {} contains only {} samples, thus using all.'
                        .format(limit_var, self.anomaly_label_strings[l], len(linclsses)), fps=False
                    )
                    new_picks.extend(linclsses)
            self.picks = sorted(new_picks)
        else:
            if limit_var is not None and limit_var < len(self):
                self.picks = np.random.choice(self.picks, size=limit_var, replace=False)
            if limit_var is not None and limit_var > len(self):
                self.logprint(
                    'OEMvTec shall be limited to {} samples, but MvTec contains only {} samples, thus using all.'
                    .format(limit_var, len(self))
                )
        if len(self) < size[0]:
            raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.picks if self.picks is not None else self.targets)

    def data_loader(self) -> DataLoader:
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        index = self.picks[index] if self.picks is not None else index

        image, label, gt = super().__getitem__(index)
        image, gt = image.mul(255).byte(), gt.mul(255).byte()

        if self.use_gt:
            return image, gt
        else:
            return image

    def logprint(self, s: str, fps=True):
        if self.logger is not None:
            self.logger.print(s, fps=fps)
        else:
            print(s)

