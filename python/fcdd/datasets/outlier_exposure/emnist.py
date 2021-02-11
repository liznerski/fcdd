import os.path as pt

import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST


def ceil(x: float):
    return int(np.ceil(x))


class MyEMNIST(EMNIST):
    """ Reimplements get_item to transform tensor input to pil image before applying transformation. """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = transforms.ToPILImage()(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class OEEMNIST(EMNIST):
    def __init__(self, size: torch.Size, root: str = None, split='letters', limit_var=20):  # split = Train
        """
        Outlier Exposure dataset for EMNIST.
        :param size: size of the samples in n x c x h x w, samples will be resized to h x w. If n is larger than the
            number of samples available in EMNIST, dataset will be enlarged by repetitions to fit n.
            This is important as exactly n images are extracted per iteration of the data_loader.
            For online supervision n should be set to 1 because only one sample is extracted at a time.
        :param root: root directory where data is found or is to be downloaded to.
        :param split: The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        :param limit_var: limits the number of different samples, i.e. randomly chooses limit_var many samples
            from all available ones to be the training data.
        """
        assert len(size) == 3 and size[1] == size[2]
        root = pt.join(root, 'emnist', )
        transform = transforms.Compose([
            transforms.Resize((size[1], size[2])),
            transforms.ToTensor()
        ])
        super().__init__(root, split, transform=transform, download=True)
        self.size = size
        self.data = self.data.transpose(1, 2)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if limit_var is not None and limit_var < len(self):
            picks = np.random.choice(np.arange(self.data.size(0)), size=limit_var, replace=False)
            self.data = self.data[picks]
            self.targets = self.targets[picks]
        if limit_var is not None and limit_var > len(self):
            print(
                'OEEMNIST shall be limited to {} samples, but EMNIST contains only {} samples, thus using all.'
                .format(limit_var, len(self))
            )
        if len(self) < size[0]:
            rep = ceil(size[0] / len(self))
            old = len(self)
            self.data = self.data.repeat(rep, 1, 1)
            self.targets = self.targets.repeat(rep)
            if rep != size[0] / old:
                import warnings
                warnings.warn(
                    'OEEMNIST has been limited to {} samples. '
                    'Due to the requested size of {}, the dataset will be enlarged. ' 
                    'But {} repetitions will make some samples appear more often than others in the dataset, '
                    'because the final size after repetitions is {}, which is cut to {}'
                    .format(limit_var, size[0], rep, len(self), size[0])
                )

    def data_loader(self) -> DataLoader:
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample, target = super().__getitem__(index)
        sample = sample.squeeze().mul(255).byte()

        return sample

