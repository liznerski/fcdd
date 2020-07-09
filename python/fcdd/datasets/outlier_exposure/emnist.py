import os.path as pt

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST


def ceil(x):
    return int(np.ceil(x))


class MyEMNIST(EMNIST):
    def __getitem__(self, index):
        """Override the original method of the EMNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
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
    def __init__(self, size, root=None, split='letters', limit_var=20):  # split = Train
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
                'OEEMNIST shall be limited to {} samples, but Cifar100 contains only {} samples, thus using all.'
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

    def data_loader(self):
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample = sample.squeeze().mul(255).byte()

        return sample

