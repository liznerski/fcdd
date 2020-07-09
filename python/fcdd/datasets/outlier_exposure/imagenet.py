import json
import os
import os.path as pt
from distutils.version import StrictVersion
from sre_constants import error as sre_constants_error

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from PIL import UnidentifiedImageError
from fcdd.datasets.confs.imagenet1k_classes import IMAGENET1k_CLS_STR
from fcdd.util import imsave
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import has_file_allowed_extension, default_loader, IMG_EXTENSIONS
from torchvision.datasets.vision import StandardTransform


def ceil(x):
    return int(np.ceil(x))


class OEImageNet(torchvision.datasets.ImageNet):
    def __init__(self, size, root=None, split='val', limit_var=1000000000, exclude=()):  # split = Train
        assert len(size) == 4 and size[2] == size[3]
        assert size[1] in [1, 3]
        assert StrictVersion(torchvision.__version__) >= StrictVersion('0.5.0')
        root = pt.join(root, 'imagenet', )
        self.root = root
        super().__init__(root, split)
        self.transform = transforms.Compose([
            transforms.Resize((size[2], size[3])),
            transforms.Grayscale() if size[1] == 1 else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])
        self.size = size
        self.picks = None
        self.picks = list(range(len(self)))
        if exclude is not None and len(exclude) > 0:
            syns = {k: v.lower().replace(' ', ',').split(',') for k, v in IMAGENET1k_CLS_STR.items()}
            exclude_ids = [i for i, s in syns.items() if any([exs.lower() in s for exs in exclude])]
            self.picks = np.argwhere(np.isin(self.targets, exclude_ids, invert=True)).flatten().tolist()
            # self.show()
            # print()
        if limit_var is not None and limit_var < len(self):
            self.picks = np.random.choice(np.arange(len(self.picks)), size=limit_var, replace=False)
        if limit_var is not None and limit_var > len(self):
            print(
                'OEImageNet shall be limited to {} samples, but ImageNet contains only {} samples, thus using all.'
                .format(limit_var, len(self))
            )
        if len(self) < size[0]:
            raise NotImplementedError()

    def __len__(self):
        return len(self.picks if self.picks is not None else self.samples)

    def data_loader(self):
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index):
        index = self.picks[index] if self.picks is not None else index

        sample, target = super().__getitem__(index)
        sample = sample.mul(255).byte()

        return sample

    def show(self, k=4):
        tars = sorted(set([self.targets[i] for i in self.picks]))
        tim = []
        for t in tars:
            limg = []
            idx = (torch.from_numpy(np.asarray(self.targets))[self.picks] == t).nonzero()[:k]
            for i in idx:
                limg.append(self[i])
            tim.append(torch.stack(limg))
        tim = torch.cat(tim)
        import pprint
        pprint.pprint(tars)
        for i in range(tim.size(0) // k):
            imsave(
                tim[i*k:i*k+k], 'imagenet1k_voc_reduced_view/{}_{}.png'.format(tars[i], IMAGENET1k_CLS_STR[tars[i]]),
                nrow=k
            )


class MyImageFolder(DatasetFolder):
    """
    Reimplements init and make_dataset. The only change is to add print lines for feedback, because
    make_dataset takes more than an hour...
    """
    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None, logger=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.metafile = os.path.join(self.root, 'meta.json')
        self.transform = transform
        self.target_transform = target_transform
        transforms = None
        if transform is not None or target_transform is not None:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        self.logger = logger
        self.loader = default_loader
        self.extensions = extensions = IMG_EXTENSIONS if is_valid_file is None else None

        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

    def make_dataset(self, dir, class_to_idx, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if os.path.exists(self.metafile):
            self.logprint('ImageFolder dataset is loading metadata from {}...'.format(self.metafile), fps=False)
            with open(self.metafile, 'r') as reader:
                images = json.load(reader)
                self.logprint('ImageFolder dataset has loaded metadata.')
        else:
            self.logprint(
                'ImageFolder dataset could not find metafile at {}. Creating it instead...'.format(self.metafile),
                fps=False
            )
            if not ((extensions is None) ^ (is_valid_file is None)):
                raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
            if extensions is not None:
                def is_valid_file(x):
                    return has_file_allowed_extension(x, extensions)
            size = len(class_to_idx.keys())
            for n, target in enumerate(sorted(class_to_idx.keys())):
                self.logprint('ImageFolder dataset is processing target {} - {}/{}'.format(target, n, size))
                d = os.path.join(dir, target)
                if not os.path.isdir(d):
                    continue
                for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if is_valid_file(path):
                            item = (path, class_to_idx[target])
                            images.append(item)
            with open(self.metafile, 'w') as writer:
                json.dump(images, writer)
        return images

    def logprint(self, s, fps=True):
        if self.logger is not None:
            self.logger.print(s, fps=fps)
        else:
            print(s)

    def logwarning(self, s, err):
        if self.logger is not None:
            self.logger.warning(s)
        else:
            raise err


class MyImageNet22K(MyImageFolder):
    imagenet1k_pairs = [
        ('acorn', 'n12267677'),
        ('airliner', 'n02690373'),
        ('ambulance', 'n02701002'),
        ('american_alligator', 'n01698640'),
        ('banjo', 'n02787622'),
        ('barn', 'n02793495'),
        ('bikini', 'n02837789'),
        ('digital_clock', 'n03196217'),
        ('dragonfly', 'n02268443'),
        ('dumbbell', 'n03255030'),
        ('forklift', 'n03384352'),
        ('goblet', 'n03443371'),
        ('grand_piano', 'n03452741'),
        ('hotdog', 'n07697537'),
        ('hourglass', 'n03544143'),
        ('manhole_cover', 'n03717622'),
        ('mosque', 'n03788195'),
        ('nail', 'n03804744'),
        ('parking_meter', 'n03891332'),
        ('pillow', 'n03938244'),
        ('revolver', 'n04086273'),
        ('rotary_dial_telephone', 'n03187595'),
        ('schooner', 'n04147183'),
        ('snowmobile', 'n04252077'),
        ('soccer_ball', 'n04254680'),
        ('stingray', 'n01498041'),
        ('strawberry', 'n07745940'),
        ('tank', 'n04389033'),
        ('toaster', 'n04442312'),
        ('volcano', 'n09472597')
    ]
    imagenet1k_idxs = [idx for name, idx in imagenet1k_pairs]

    def __init__(self, root, size, exclude_imagenet1k=True, *args, **kwargs):
        super(MyImageNet22K, self).__init__(root, *args, **kwargs)

        self.exclude_imagenet1k = exclude_imagenet1k
        self.shuffle_idxs = False
        self.size = size

        if exclude_imagenet1k:
            self.samples = [s for s in self.samples if not any([idx in s[0] for idx in self.imagenet1k_idxs])]

    def __getitem__(self, index):
        """Override the original method of the ImageFolder class to catch some errors (seems like a few of the 22k
        images are broken).
        :return tuple: (sample, target)
        """
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except UnidentifiedImageError as e:
            msg = 'ImageNet22k could not load picture at {}. Unidentified image error.'.format(path)
            self.logwarning(msg, e)
            sample = transforms.ToPILImage()(torch.zeros(self.size[1:]).byte())
        except OSError as e:
            msg = 'ImageNet22k could not load picture at {}. OS Error.'.format(path)
            self.logwarning(msg, e)
            sample = transforms.ToPILImage()(torch.zeros(self.size[1:]).byte())
        except sre_constants_error as e:
            msg = 'ImageNet22k could not load picture at {}. SRE Constants Error.'.format(path)
            self.logwarning(msg, e)
            sample = transforms.ToPILImage()(torch.zeros(self.size[1:]).byte())
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, 1


class OEImageNet22k(MyImageNet22K):
    def __init__(self, size, root=None, limit_var=1000000000, augment=False, logger=None):  # split = Train
        assert len(size) == 4 and size[2] == size[3]
        assert size[1] in [1, 3]
        assert StrictVersion(torchvision.__version__) >= StrictVersion('0.5.0')
        assert not augment, 'Need to fix size in MyImageNet22k for this, as it expects size to be the raw size...'
        root = pt.join(root, 'imagenet22k') if not root.endswith('imagenet') else pt.join(root, '..', 'imagenet22k')
        root = pt.join(root, 'fall11_whole_extracted')  # important to have a second layer, to speed up load meta file
        self.root = root
        self.augment = augment
        self.logger = logger
        with logger.timeit('Loading ImageNet22k'):
            super().__init__(root=root, size=size, logger=logger)

        tnon = transforms.Lambda(lambda x: x)
        self.transform = transforms.Compose([
            transforms.Resize(int(size[2] * 1.15)) if augment else transforms.Resize(size[2]),
            transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01) if augment else tnon,
            transforms.RandomHorizontalFlip(p=0.5) if augment else tnon,
            transforms.RandomCrop(size[2]) if augment else tnon,
            transforms.ToTensor()
        ])
        self.picks = None
        if limit_var is not None and limit_var < len(self):
            self.picks = np.random.choice(len(self.samples), size=limit_var, replace=False)
        if limit_var is not None and limit_var > len(self):
            self.logprint(
                'OEImageNet22 shall be limited to {} samples, but ImageNet22k contains only {} samples, thus using all.'
                .format(limit_var, len(self)), fps=False
            )
        if len(self) < size[0]:
            raise NotImplementedError()

    def __len__(self):
        return len(self.picks if self.picks is not None else self.samples)

    def data_loader(self):
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index):
        index = self.picks[index] if self.picks is not None else index

        sample, target = super().__getitem__(index)
        sample = sample.mul(255).byte()

        return sample

