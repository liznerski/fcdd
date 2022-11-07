import os
import shutil
import tarfile
import tempfile
from typing import Callable
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from fcdd.datasets.bases import GTMapADDataset
from fcdd.util.logging import Logger
from torchvision.datasets import VisionDataset
from torchvision.datasets.imagenet import check_integrity, verify_str_arg


class MvTec(VisionDataset, GTMapADDataset):
    """ Implemention of a torch style MVTec dataset """
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    base_folder = 'mvtec'
    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )
    normal_anomaly_label = 'good'
    normal_anomaly_label_idx = 0

    def __init__(self, root: str, split: str = 'train', target_transform: Callable = None,
                 img_gt_transform: Callable = None, transform: Callable = None, all_transform: Callable = None,
                 download=True, shape=(3, 300, 300), normal_classes=(), nominal_label=0, anomalous_label=1,
                 logger: Logger = None, enlarge: bool = False
                 ):
        """
        Loads all data from the prepared torch tensors. If such torch tensors containg MVTec data are not found
        in the given root directory, instead downloads the raw data and prepares the tensors.
        They contain labels, images, and ground-truth maps for a fixed size, determined by the shape parameter.
        :param root: directory where the data is to be found.
        :param split: whether to use "train", "test", or "test_anomaly_label_target" data.
            In the latter case the get_item method returns labels indexing the anomalous class rather than
            the object class. That is, instead of returning 0 for "bottle", it returns "1" for "large_broken".
        :param target_transform: function that takes label and transforms it somewhat.
            Target transform is the first transform that is applied.
        :param img_gt_transform: function that takes image and ground-truth map and transforms it somewhat.
            Useful to apply the same augmentation to image and ground-truth map (e.g. cropping), s.t.
            the ground-truth map still matches the image.
            ImgGt transform is the third transform that is applied.
        :param transform: function that takes image and transforms it somewhat.
            Transform is the last transform that is applied.
        :param all_transform: function that takes image, label, and ground-truth map and transforms it somewhat.
            All transform is the second transform that is applied.
        :param download: whether to download if data is not found in root.
        :param shape: the shape (c x h x w) the data should be resized to (images and ground-truth maps).
        :param normal_classes: all the classes that are considered nominal (usually just one).
        :param nominal_label: the label that is to be returned to mark nominal samples.
        :param anomalous_label: the label that is to be returned to mark anomalous samples.
        :param logger: logger
        :param enlarge: whether to enlarge the dataset, i.e. repeat all data samples ten times.
            Consequently, one iteration (epoch) of the data loader returns ten times as many samples.
            This speeds up loading because the MVTec-AD dataset has a poor number of samples and
            PyTorch requires additional work in between epochs.
        """
        super(MvTec, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", ("train", "test", "test_anomaly_label_target"))
        self.img_gt_transform = img_gt_transform
        self.all_transform = all_transform
        self.shape = shape
        self.orig_gtmaps = None
        self.normal_classes = normal_classes
        self.nominal_label = nominal_label
        self.anom_label = anomalous_label
        self.logger = logger
        self.enlarge = enlarge

        if download:
            self.download(shape=self.shape[1:])

        print('Loading dataset from {}...'.format(self.data_file))
        dataset_dict = torch.load(self.data_file)
        self.anomaly_label_strings = dataset_dict['anomaly_label_strings']
        if self.split == 'train':
            self.data, self.targets = dataset_dict['train_data'], dataset_dict['train_labels']
            self.gt, self.anomaly_labels = None, None
        else:
            self.data, self.targets = dataset_dict['test_data'], dataset_dict['test_labels']
            self.gt, self.anomaly_labels = dataset_dict['test_maps'], dataset_dict['test_anomaly_labels']

        if self.enlarge:
            self.data, self.targets = self.data.repeat(10, 1, 1, 1), self.targets.repeat(10)
            self.gt = self.gt.repeat(10, 1, 1) if self.gt is not None else None
            self.anomaly_labels = self.anomaly_labels.repeat(10) if self.anomaly_labels is not None else None
            self.orig_gtmaps = self.orig_gtmaps.repeat(10, 1, 1) if self.orig_gtmaps is not None else None

        if self.nominal_label != 0:
            print('Swapping labels, i.e. anomalies are 0 and nominals are 1, same for GT maps.')
            assert -3 not in [self.nominal_label, self.anom_label]
        print('Dataset complete.')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img, label = self.data[index], self.targets[index]

        if self.split == 'test_anomaly_label_target':
            label = self.target_transform(self.anomaly_labels[index])
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.split == 'train' and self.gt is None:
            assert self.anom_label in [0, 1]
            # gt is assumed to be 1 for anoms always (regardless of the anom_label), since the supervisors work that way
            # later code fixes that (and thus would corrupt it if the correct anom_label is used here in swapped case)
            gtinitlbl = label if self.anom_label == 1 else (1 - label)
            gt = (torch.ones_like(img)[0] * gtinitlbl).mul(255).byte()
        else:
            gt = self.gt[index]

        if self.all_transform is not None:
            img, gt, label = self.all_transform((img, gt, label))
            gt = gt.mul(255).byte() if gt.dtype != torch.uint8 else gt
            img = img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else img

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(0, 2).transpose(0, 1).numpy(), mode='RGB')
        gt = Image.fromarray(gt.squeeze(0).numpy(), mode='L')

        if self.img_gt_transform is not None:
            img, gt = self.img_gt_transform((img, gt))

        if self.transform is not None:
            img = self.transform(img)

        if self.nominal_label != 0:
            gt[gt == 0] = -3  # -3 is chosen arbitrarily here
            gt[gt == 1] = self.anom_label
            gt[gt == -3] = self.nominal_label

        return img, label, gt

    def __len__(self) -> int:
        return len(self.data)

    def download(self, verbose=True, shape=None, cls=None):
        assert shape is not None or cls is not None, 'original shape requires a class'
        if not check_integrity(self.data_file if shape is not None else self.orig_data_file(cls)):
            tmp_dir = tempfile.mkdtemp()
            self.download_and_extract_archive(
                self.url, os.path.join(self.root, self.base_folder), extract_root=tmp_dir,
            )
            train_data, train_labels = [], []
            test_data, test_labels, test_maps, test_anomaly_labels = [], [], [], []
            anomaly_labels, albl_idmap = [], {self.normal_anomaly_label: self.normal_anomaly_label_idx}

            for lbl_idx, lbl in enumerate(self.labels if cls is None else [self.labels[cls]]):
                if verbose:
                    print('Processing data for label {}...'.format(lbl))
                for anomaly_label in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'test'))):
                    for img_name in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'test', anomaly_label))):
                        with open(os.path.join(tmp_dir, lbl, 'test', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        if anomaly_label != self.normal_anomaly_label:
                            mask_name = self.convert_img_name_to_mask_name(img_name)
                            with open(os.path.join(tmp_dir, lbl, 'ground_truth', anomaly_label, mask_name), 'rb') as f:
                                mask = Image.open(f)
                                mask = self.img_to_torch(mask, shape)
                        else:
                            mask = torch.zeros_like(sample)
                        test_data.append(sample)
                        test_labels.append(cls if cls is not None else lbl_idx)
                        test_maps.append(mask)
                        if anomaly_label not in albl_idmap:
                            albl_idmap[anomaly_label] = len(albl_idmap)
                        test_anomaly_labels.append(albl_idmap[anomaly_label])

                for anomaly_label in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'train'))):
                    for img_name in sorted(os.listdir(os.path.join(tmp_dir, lbl, 'train', anomaly_label))):
                        with open(os.path.join(tmp_dir, lbl, 'train', anomaly_label, img_name), 'rb') as f:
                            sample = Image.open(f)
                            sample = self.img_to_torch(sample, shape)
                        train_data.append(sample)
                        train_labels.append(lbl_idx)

            anomaly_labels = list(zip(*sorted(albl_idmap.items(), key=lambda kv: kv[1])))[0]
            train_data = torch.stack(train_data)
            train_labels = torch.IntTensor(train_labels)
            test_data = torch.stack(test_data)
            test_labels = torch.IntTensor(test_labels)
            test_maps = torch.stack(test_maps)[:, 0, :, :]  # r=g=b -> grayscale
            test_anomaly_labels = torch.IntTensor(test_anomaly_labels)
            torch.save(
                {'train_data': train_data, 'train_labels': train_labels,
                 'test_data': test_data, 'test_labels': test_labels,
                 'test_maps': test_maps, 'test_anomaly_labels': test_anomaly_labels,
                 'anomaly_label_strings': anomaly_labels},
                self.data_file if shape is not None else self.orig_data_file(cls)
            )

            # cleanup temp directory
            for dirpath, dirnames, filenames in os.walk(tmp_dir):
                os.chmod(dirpath, 0o755)
                for filename in filenames:
                    os.chmod(os.path.join(dirpath, filename), 0o755)
            shutil.rmtree(tmp_dir)
        else:
            print('Files already downloaded.')
            return

    def get_original_gtmaps_normal_class(self) -> torch.Tensor:
        """
        Returns ground-truth maps of original size for test samples.
        The class is chosen according to the normal class the dataset was created with.
        This method is usually used for pixel-wise ROC computation.
        """
        assert self.split != 'train', 'original maps are only available for test mode'
        assert len(self.normal_classes) == 1, 'normal classes must be known and there must be exactly one'
        assert self.all_transform is None, 'all_transform would be skipped here'
        assert all([isinstance(t, (transforms.Resize, transforms.ToTensor)) for t in self.img_gt_transform.transforms])
        if self.orig_gtmaps is None:
            self.download(shape=None, cls=self.normal_classes[0])
            orig_ds = torch.load(self.orig_data_file(self.normal_classes[0]))
            self.orig_gtmaps = orig_ds['test_maps'].unsqueeze(1).div(255)
        return self.orig_gtmaps

    @property
    def data_file(self):
        return os.path.join(self.root, self.base_folder, self.filename)

    @property
    def filename(self):
        return "admvtec_{}x{}.pt".format(self.shape[1], self.shape[2])

    def orig_data_file(self, cls):
        return os.path.join(self.root, self.base_folder, self.orig_filename(cls))

    def orig_filename(self, cls):
        return "admvtec_orig_cls{}.pt".format(cls)

    @staticmethod
    def img_to_torch(img, shape=None):
        if shape is not None:
            return torch.nn.functional.interpolate(
                torch.from_numpy(np.array(img.convert('RGB'))).float().transpose(0, 2).transpose(1, 2)[None, :],
                shape
            )[0].byte()
        else:
            return torch.from_numpy(
                np.array(img.convert('RGB'))
            ).float().transpose(0, 2).transpose(1, 2)[None, :][0].byte()

    @staticmethod
    def convert_img_name_to_mask_name(img_name):
        return img_name.replace('.png', '_mask.png')

    @staticmethod
    def download_and_extract_archive(url, download_root, extract_root=None, filename=None, remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)
        if not os.path.exists(download_root):
            os.makedirs(download_root)

        MvTec.download_url(url, download_root, filename)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        MvTec.extract_archive(archive, extract_root, remove_finished)

    @staticmethod
    def download_url(url, root, filename=None):
        """Download a file from a url and place it in root.
        Args:
            url (str): URL to download file from
            root (str): Directory to place downloaded file in
            filename (str, optional): Name to save the file under. If None, use the basename of the URL
        """
        from six.moves import urllib

        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)

        def gen_bar_updater():
            pbar = tqdm(total=None)

            def bar_update(count, block_size, total_size):
                if pbar.total is None and total_size:
                    pbar.total = total_size
                progress_bytes = count * block_size
                pbar.update(progress_bytes - pbar.n)

            return bar_update

        # check if file is already present locally
        if not check_integrity(fpath, None):
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(
                        url, fpath,
                        reporthook=gen_bar_updater()
                    )
                else:
                    raise e
            # check integrity of downloaded file
            if not check_integrity(fpath, None):
                raise RuntimeError("File not found or corrupted.")

    @staticmethod
    def extract_archive(from_path, to_path=None, remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        with tarfile.open(from_path, 'r:xz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
