import os.path as pt

import numpy as np
import torchvision.transforms as transforms
from fcdd.datasets.mvtec_base import MvTec
from fcdd.datasets.preprocessing import MultiCompose, get_target_label_idx
from torch.utils.data import DataLoader


def ceil(x):
    return int(np.ceil(x))


class OEMvTec(MvTec):
    def __init__(self, size, clsses, root=None, limit_var=100000000, limit_per_anomaly=True,
                 download=True, logger=None, gt=False, remove_nominal=True):
        assert len(size) == 4 and size[2] == size[3]
        assert size[1] in [1, 3]
        self.root = root
        self.logger = logger
        self.size = size
        self.use_gt = gt
        self.clsses = clsses
        super().__init__(root, 'test', download=download, shape=size[1:], logger=logger)

        self.img_gt_transform = MultiCompose([
            transforms.Resize(size[2]),
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

    def __len__(self):
        return len(self.picks if self.picks is not None else self.targets)

    def data_loader(self):
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index):
        index = self.picks[index] if self.picks is not None else index

        image, label, gt = super().__getitem__(index)
        image, gt = image.mul(255).byte(), gt.mul(255).byte()

        if self.use_gt:
            return image, gt
        else:
            return image

    def logprint(self, s, fps=True):
        if self.logger is not None:
            self.logger.print(s, fps=fps)
        else:
            print(s)

