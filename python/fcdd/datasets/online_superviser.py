import random
import traceback
from fcdd.datasets.preprocessing import ImgGTTargetTransform
from fcdd.datasets.outlier_exposure.imagenet import OEImageNet, OEImageNet22k
from fcdd.datasets.outlier_exposure.cifar100 import OECifar100
from fcdd.datasets.outlier_exposure.emnist import OEEMNIST
from fcdd.datasets.outlier_exposure.mvtec import OEMvTec
from itertools import cycle


class OnlineSuperviser(ImgGTTargetTransform):
    def __init__(self, ds, supervise_mode, supervise_params, p=0.5):
        self.ds = ds
        self.supervise_mode = supervise_mode
        self.supervise_params = supervise_params
        self.p = p
        self.noise_sampler = None
        if self.supervise_params.get('noise_mode', '') == 'imagenet':
            self.noise_sampler = cycle(
                OEImageNet(
                    (1, ) + ds.raw_shape, limit_var=supervise_params.get('limit', None), root=ds.root
                ).data_loader()
            )
        elif self.supervise_params.get('noise_mode', '') == 'imagenet_for_voc':
            self.noise_sampler = cycle(
                OEImageNet(
                    (1, ) + ds.raw_shape, limit_var=supervise_params.get('limit', None),
                    exclude=self.supervise_params['exclude_for_voc'], root=ds.root
                ).data_loader()
            )
        elif self.supervise_params.get('noise_mode', '') == 'imagenet22k':
            self.noise_sampler = cycle(
                OEImageNet22k(
                    (1, ) + ds.raw_shape, limit_var=supervise_params.get('limit', None), logger=ds.logger,
                    root=ds.root
                ).data_loader()
            )
        elif self.supervise_params.get('noise_mode', '') == 'cifar100':
            self.noise_sampler = cycle(
                OECifar100(
                    (1, ) + ds.raw_shape, limit_var=supervise_params.get('limit', None),
                    root=ds.root
                ).data_loader(),
            )
        elif self.supervise_params.get('noise_mode', '') == 'emnist':
            self.noise_sampler = cycle(
                OEEMNIST(
                    (1, ) + ds.raw_shape, limit_var=supervise_params.get('limit', None),
                    root=ds.root
                ).data_loader()
            )
        elif self.supervise_params.get('noise_mode', '') == 'mvtec':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, ds.normal_classes, limit_var=supervise_params.get('limit', None),
                    logger=ds.logger, root=ds.root
                ).data_loader()
            )
        elif self.supervise_params.get('noise_mode', '') == 'mvtec_gt':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, ds.normal_classes, limit_var=supervise_params.get('limit', None),
                    logger=ds.logger, gt=True, root=ds.root
                ).data_loader()
            )
        elif self.supervise_params.get('noise_mode', '') == 'mvtec_gt_randomdefections':
            self.noise_sampler = cycle(
                OEMvTec(
                    (1, ) + ds.raw_shape, ds.normal_classes, limit_var=supervise_params.get('limit', None),
                    logger=ds.logger, gt=True, limit_per_anomaly=False, root=ds.root
                ).data_loader()
            )

    def __call__(self, img, gt, target, replace=None):
        active = self.supervise_mode not in ['other', 'unsupervised']
        if active and (replace or replace is None and random.random() < self.p):
            supervise_mode = self.supervise_mode
            img = img.unsqueeze(0) if img is not None else img
            # gt value 1 will be put to anom_label in mvtec_bases get_item
            gt = gt.unsqueeze(0).unsqueeze(0).fill_(1).float() if gt is not None else gt
            if self.noise_sampler is None:
                generated_noise = self.ds._generate_noise(
                    self.supervise_params.get('noise_mode', None), img.shape
                )
            else:
                try:
                    generated_noise = next(self.noise_sampler)
                except RuntimeError:
                    generated_noise = next(self.noise_sampler)
                    self.ds.logger.warning(
                        'Had to resample in online_superviser __call__ next(self.noise_sampler) because of {}'
                        .format(traceback.format_exc())
                    )
                if isinstance(generated_noise, (tuple, list)):
                    generated_noise, gt = generated_noise
            if supervise_mode in ['noise']:
                img, gt, target = self.noise(img, gt, target, self.ds, generated_noise)
            elif supervise_mode in ['malformed_normal']:
                img, gt, target = self.malformed_normal(img, gt, target, self.ds, generated_noise)
            elif supervise_mode in ['malformed_normal_gt']:
                img, gt, target = self.malformed_normal(img, gt, target, self.ds, generated_noise, use_gt=True)
            else:
                raise NotImplementedError('Supervise mode {} unknown.'.format(supervise_mode))
            img = img.squeeze(0) if img is not None else img
            gt = gt.squeeze(0).squeeze(0) if gt is not None else gt
        return img, gt, target

    def noise(self, img, gt, target, ds, generated_noise, use_gt=False):
        if use_gt:
            raise ValueError('No GT mode for pure noise available!')
        anom = generated_noise.clamp(0, 255).byte()
        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied
        return anom, gt, t

    def malformed_normal(self, img, gt, target, ds, generated_noise, use_gt=False, brightness_threshold=0.11*255):
        assert (img.dim() == 4 or img.dim() == 3) and generated_noise.shape == img.shape
        anom = img.clone()

        # invert noise for bright regions (bright regions are considered being on average > 0.33 * 255)
        generated_noise = generated_noise.int()
        bright_regions = img.sum(1) > brightness_threshold * img.shape[1]
        for ch in range(img.shape[1]):
            gnch = generated_noise[:, ch]
            gnch[bright_regions] = gnch[bright_regions] * -1
            generated_noise[:, ch] = gnch

        anom = (anom.int() + generated_noise).clamp(0, 255).byte()

        t = 1 if not hasattr(ds, 'anomalous_label') else ds.anomalous_label  # target transform has already been applied

        if use_gt:
            gt = (img != anom).max(1)[0].clone().float()
            gt = gt.unsqueeze(1)  # value 1 will be put to anom_label in mvtec_bases get_item

        return anom, gt, t
