import random
import os
import os.path as pt
import numpy as np
import torchvision.transforms as transforms
import torch
import PIL.Image as Image
from typing import Tuple, List
from torch import Tensor
from torch.nn.functional import interpolate
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension
from fcdd.datasets.bases import GTMapADDataset, GTSubset
from fcdd.datasets.preprocessing import get_target_label_idx, MultiCompose
from fcdd.datasets.image_folder import ADImageFolderDataset, ImageFolderDataset
from fcdd.util.logging import Logger


def extract_custom_classes(datapath: str) -> List[str]:
    dir = os.path.join(datapath, 'custom', 'test')
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    return classes


class ADImageFolderDatasetGTM(ADImageFolderDataset):

    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool,
                 logger: Logger = None):
        """
        This is a general-purpose implementation for custom datasets.
        It expects the data being contained in class folders and distinguishes between
        (1) the one-vs-rest (ovr) approach where one class is considered normal
        and is tested against all other classes being anomalous
        (2) the general approach where each class folder contains a normal data folder and an anomalous data folder.
        The :attr:`ovr` determines this.

        For (1) the data folders have to follow this structure:
        root/custom/train/dog/xxx.png
        root/custom/train/dog/xxy.png
        root/custom/train/dog/xxz.png

        root/custom/train/cat/123.png
        root/custom/train/cat/nsdf3.png
        root/custom/train/cat/asd932_.png

        For (2):
        root/custom/train/hazelnut/normal/xxx.png
        root/custom/train/hazelnut/normal/xxy.png
        root/custom/train/hazelnut/normal/xxz.png
        root/custom/train/hazelnut/anomalous/xxa.png    -- may be used during training for a semi-supervised setting

        root/custom/train/screw/normal/123.png
        root/custom/train/screw/normal/nsdf3.png
        root/custom/train/screw/anomalous/asd932_.png   -- may be used during training for a semi-supervised setting

        The same holds for the test set, where "train" has to be replaced by "test".

        To take advantage of available binary ground-truth anomaly maps, you need to place them in separate folders.
        That is, create the folders "train_maps" and/or "test_maps" and place the corresponding maps using the same structure
        and name as above. For instance:
        root/custom/train_maps/dog/xxx.png    for    root/custom/train/dog/xxx.png or
        root/custom/test_maps/screw/normal/123.png   for   root/custom/test/screw/normal/123.png

        The ground-truth maps need to be binary; i.e., need to be in {0, 255}^{1 x h x w}, where 255 marks anomalous regions.
        Missing maps are replaced by tensors filled with the corresponding label (e.g., 255 for anomalies).
        That is, a completely white or black image.
        However, for computing a pixel-wise ROC score measuring the explanation performance, all maps for the anomalous
        test samples are required. Otherwise, the pixel-wise ROC evaluation is skipped.


        :param root: root directory where data is found.
        :param normal_class: the class considered normal.
        :param preproc: the kind of preprocessing pipeline.s
        :param nominal_label: the label that marks normal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the normal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode).
        :param online_supervision: whether to sample anomalies online in each epoch,
            or offline before training (same for all epochs in this case).
        :param logger: logger.
        """
        super().__init__(
            root, normal_class, preproc, nominal_label, supervise_mode, noise_mode, oe_limit, online_supervision, logger
        )
        self.check_gtm_data()  # you may remove this line to speed up dataset loading

        # img_gtm transforms transform images and corresponding ground-truth maps jointly.
        # This is critically required for random geometric transformations as otherwise
        # the maps would not match the images anymore.
        if preproc in ['', None, 'default', 'none']:
            img_gtm_test_transform = img_gtm_transform = MultiCompose([
                transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST),
                transforms.ToTensor(),
            ])
            test_transform = transform = transforms.Compose([
                transforms.Normalize(self.mean, self.std)
            ])
        elif preproc in ['aug1']:
            img_gtm_transform = MultiCompose([
                transforms.RandomChoice([
                    MultiCompose([
                        transforms.Resize((self.raw_shape[-2], self.raw_shape[-1]), Image.NEAREST),
                        transforms.RandomCrop((self.shape[-2], self.shape[-1]), Image.NEAREST),
                    ]),
                    transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST),
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            img_gtm_test_transform = MultiCompose(
                [transforms.Resize((self.shape[-2], self.shape[-1]), Image.NEAREST), transforms.ToTensor()]
            )
            test_transform = transforms.Compose([
                transforms.Normalize(self.mean, self.std)
            ])
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
                transforms.Normalize(self.mean, self.std)
            ])
        #  here you could define other pipelines with augmentations
        else:
            raise ValueError('Preprocessing pipeline {} is not known.'.format(preproc))

        self._train_set = ImageFolderDatasetGTM(
            self.trainpath, supervise_mode, self.raw_shape, self.ovr, self.nominal_label, self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=transform, target_transform=self.target_transform, all_transform=self.all_transform,
            img_gtm_transform=img_gtm_transform
        )
        if supervise_mode == 'other':  # (semi)-supervised setting
            self.balance_dataset(gtm=True)
        else:
            self._train_set = GTSubset(
                self._train_set, np.argwhere(
                    (np.asarray(self._train_set.anomaly_labels) == self.nominal_label) *
                    np.isin(self._train_set.targets, self.normal_classes)
                ).flatten().tolist()
            )

        self._test_set = ImageFolderDatasetGTM(
            self.testpath, supervise_mode, self.raw_shape, self.ovr, self.nominal_label, self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=test_transform, target_transform=self.target_transform,
            img_gtm_transform=img_gtm_test_transform
        )
        if self.ovr:
            self._test_set = GTSubset(
                self._test_set, np.arange(len(self.test_set.targets))
            )
        else:
            self._test_set = GTSubset(
                self._test_set, get_target_label_idx(self._test_set.targets, np.asarray(self.normal_classes))
            )

    def check_gtm_data(self):
        if self.gtm:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, IMG_EXTENSIONS)  # type: ignore[arg-type]
            for split_dir in (self.trainpath, self.testpath):
                gtm_split_dir = f"{split_dir}_maps"
                if pt.exists(gtm_split_dir):
                    for cls_dir in os.listdir(gtm_split_dir):
                        if cls_dir not in os.listdir(split_dir):
                            raise ValueError(f'Class {cls_dir} found in {gtm_split_dir} but not in {split_dir}!')
                        for lbl_dir in os.listdir(pt.join(gtm_split_dir, cls_dir)) if not self.ovr else ('.', ):
                            if not self.ovr and lbl_dir.lower() not in ('normal', 'anomalous', 'nominal'):
                                raise ValueError(
                                    f'All class folders need to contain folders for "normal" and "anomalous" data. '
                                    f'However, found a folder named {lbl_dir} in {pt.join(gtm_split_dir, cls_dir)}.'
                                )
                            for img in os.listdir(pt.join(gtm_split_dir, cls_dir, lbl_dir)):
                                if is_valid_file(img):
                                    if not pt.exists(pt.join(split_dir, cls_dir, lbl_dir, img)):
                                        raise ValueError(
                                            f'There is a ground-truth map named {img} in '
                                            f'{pt.join(gtm_split_dir, cls_dir, lbl_dir)}, but there is no corresponding image '
                                            f'in {pt.join(split_dir, cls_dir, lbl_dir)} !'
                                        )


class ImageFolderDatasetGTM(ImageFolderDataset, GTMapADDataset):
    def __init__(self, root: str, supervise_mode: str, raw_shape: Tuple[int, int, int], ovr: bool,
                 nominal_label: int, anomalous_label: int,
                 transform=None, target_transform=None, normal_classes=None, all_transform=None, img_gtm_transform=None):
        super().__init__(
            root, supervise_mode, raw_shape, ovr, nominal_label, anomalous_label, transform, target_transform,
            normal_classes, all_transform
        )
        self.nominal_label = nominal_label
        self.anomalous_label = anomalous_label
        self.img_gtm_transform = img_gtm_transform
        gtmroot = f"{self.root}_maps"
        self.gtm_samples = [
            (path.replace(self.root, gtmroot), t) if pt.exists(path.replace(self.root, gtmroot)) else (None, t)
            for (path, _), t in zip(self.samples, self.anomaly_labels)
        ]

    def __getitem__(self, index: int) -> Tuple[Tensor, int, Tensor]:
        target = self.anomaly_labels[index]
        gt = None

        if self.target_transform is not None:
            pass  # already applied since we use self.anomaly_labels instead of self.targets

        if self.all_transform is not None:
            replace = random.random() < 0.5
            if replace:
                if self.supervise_mode not in ['malformed_normal', 'malformed_normal_gt']:
                    img, _, target = self.all_transform(torch.empty(self.raw_shape), None, target, replace=replace)
                else:
                    path, _ = self.samples[index]
                    img = to_tensor(self.loader(path)).mul(255).byte()
                    img, gt, target = self.all_transform(img, None, target, replace=replace)
                img = to_pil_image(img)
                gt = gt.mul(255).byte() if gt is not None and gt.dtype != torch.uint8 else gt
                gt = to_pil_image(gt) if gt is not None else None
            else:
                path, _ = self.samples[index]
                gt_path, _ = self.gtm_samples[index]
                img = self.loader(path)
                if gt_path is not None:
                    gt = self.loader(gt_path)
        else:
            path, _ = self.samples[index]
            gt_path, _ = self.gtm_samples[index]
            img = self.loader(path)
            if gt_path is not None:
                gt = self.loader(gt_path)

        if gt is None:
            # gt is assumed to be 1 for anoms always (regardless of the anom_label), since the supervisors work that way
            # later code fixes that (and thus would corrupt it if the correct anom_label is used here in swapped case)
            gtinitlbl = target if self.anomalous_label == 1 else (1 - target)
            gt = (torch.ones(self.raw_shape)[0] * gtinitlbl).mul(255).byte()
            gt = to_pil_image(gt)

        if self.img_gtm_transform is not None:
            img, gt = self.img_gtm_transform((img, gt))

        if self.transform is not None:
            img = self.transform(img)

        if self.nominal_label != 0:
            gt[gt == 0] = -3  # -3 is chosen arbitrarily here
            gt[gt == 1] = self.anomalous_label
            gt[gt == -3] = self.nominal_label

        gt = gt[:1]  # cut off redundant channels

        return img, target, gt

    def get_original_gtmaps_normal_class(self) -> torch.Tensor:
        """
        Returns ground-truth maps of original size for test samples.
        The class is chosen according to the normal class the dataset was created with.
        This method is usually used for pixel-wise ROC computation.
        """
        assert len(self.normal_classes) == 1, 'Normal classes must be known and there must be exactly one.'
        assert self.all_transform is None, 'All_transform would be skipped here.'
        assert all([isinstance(t, (transforms.Resize, transforms.ToTensor)) for t in self.img_gtm_transform.transforms]), \
            "If other transforms than resize are used, the original-sized ground-truth maps do not match the heatmaps. "
        orig_gtmaps = [
            to_tensor(self.loader(g)) if g is not None
            else ((torch.ones(self.raw_shape) * self.nominal_label) if albl == self.nominal_label else None)
            for (g, albl), t in zip(self.gtm_samples, self.targets) if t == self.normal_classes[0]
        ]
        assert all([g is not None for g in orig_gtmaps]), 'For some samples no ground-truth maps were found.'
        minsize = min([min(g.shape[-2:]) for g in orig_gtmaps])
        orig_gtmaps = torch.cat([interpolate(g.unsqueeze(0), (minsize, minsize), mode='nearest') for g in orig_gtmaps])[:, :1]
        return orig_gtmaps
