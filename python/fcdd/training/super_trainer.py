from typing import List, Tuple

import torch
from fcdd.models.bases import FCDDNet, BaseNet
from fcdd.training.ae import AETrainer
from fcdd.training.fcdd import FCDDTrainer
from fcdd.training.hsc import HSCTrainer
from fcdd.util.logging import Logger
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader


class SuperTrainer(object):
    def __init__(
            self, net: BaseNet, opt: Optimizer, sched: _LRScheduler, dataset_loaders: Tuple[DataLoader, DataLoader],
            logger: Logger, device=torch.device('cuda:0'), objective='fcdd',
            quantile=0.93, resdown=64, gauss_std: float = None, blur_heatmaps=True
    ):
        """
        This super trainer maintains networks, optimizers, schedulers, and everything related to training and testing.
        Most importantely, it offers the :meth:`train` and :meth:`test`.
        Both methods are adjusted to fit the objective, i.e. the super trainer creates individual sub trainer instances
        based on the objective -- e.g. an FCDDTrainer for the FCDD objective -- whose train and test methods are invoked
        respectively.
        For training, the trainer trains the network using the optimizer and scheduler, and it
        logs losses and other relevant metrics.
        For testing, the trainer computes scores (ROCs) for the test samples and generates heatmaps for samples
        arranged in different ways for both training and test sets (see :meth:`fcdd.training.bases.BaseADTrainer.test`)

        :param net: neural network model that is to be trained and tested.
        :param opt: some optimizer that is used to fit the network parameters.
        :param sched: some learning rate scheduler that adjusts the learning rate during training.
        :param dataset_loaders: train and test loader, might either return 2 values (input, label)
            or 3 values (input, label, ground-truth map).
        :param logger: some logger that is used to log all training and test information.
        :param device: either a cuda device or cpu.
        :param objective: the objective that is to be trained, one of {'fcdd', 'hsc', 'ae'}.
        :param quantile: the quantile used for normalizing the heatmaps (see Appendix of the paper).
        :param resdown: the maximum allowed resolution per heatmap for logged images (height and width at once).
        :param gauss_std: the standard deviation used for Gaussian kernels (upsampling and blurring).
            None defaults to the formula in :func:`fcdd.datasets.noise.kernel_size_to_std`.
        :param blur_heatmaps: whether to blur heatmaps that have not been upsampled with a Gaussian kernel.
        """
        if objective in ['fcdd']:
            assert isinstance(net, FCDDNet), 'For the FCDD objective, the net needs to be an FCDD net!'
        elif objective in ['hsc']:
            assert not isinstance(net, FCDDNet), 'For the HSC objective, the net must not be an FCDD net!'
        elif objective in ['ae']:
            assert hasattr(net, 'encoder_cls'), 'For the AE objective, the net must be an autoencoder!'

        if objective == 'fcdd':
            self.trainer = FCDDTrainer(
                net, opt, sched, dataset_loaders, logger, objective, gauss_std, quantile, resdown, blur_heatmaps, device
            )
        elif objective == 'hsc':
            self.trainer = HSCTrainer(
                net, opt, sched, dataset_loaders, logger, objective, gauss_std, quantile, resdown, blur_heatmaps, device
            )
        else:
            self.trainer = AETrainer(
                net, opt, sched, dataset_loaders, logger, objective, gauss_std, quantile, resdown, blur_heatmaps, device
            )

        self.logger = logger
        self.res = {}  # keys = {pt_roc, roc, gtmap_roc, prc, gtmap_prc}

    def train(self, epochs: int, snap: str = None, acc_batches=1):
        """
        Trains the model for anomaly detection. Afterwards, stores and plots all metrics that have
        been logged during training in respective files in the log directory. Additionally, saves a snapshot
        of the model.
        :param epochs: number of epochs (full data loader iterations).
        :param snap: path to training snapshot to load network parameters for model before any training.
            If epochs exceeds the current epoch loaded from the snapshot, training is continued with
            the optimizer and schedulers having loaded their state from the snapshot as well.
        :param acc_batches: accumulate that many batches (see :meth:`fcdd.training.bases.BaseTrainer.train`).
        """
        start = self.load(snap)

        try:
            self.trainer.train(epochs - start, acc_batches)
        finally:
            self.logger.save()
            self.logger.plot()
            self.trainer.snapshot(epochs)

    def test(self, specific_viz_ids: Tuple[List[int], List[int]] = ()) -> dict:
        """
        Tests the model, i.e. computes scores and heatmaps and stores them in the log directory.
        For details see :meth:`fcdd.training.bases.BaseTrainer.test`.
        :param specific_viz_ids: See :meth:`fcdd.training.bases.BaseTrainer.test`
        """
        res = self.trainer.test(specific_viz_ids)
        if res is not None:
            self.res.update(res)
        return self.res

    def load(self, path):
        """ loads snapshot of model parameters and training state """
        epoch = 0
        if path is not None:
            epoch = self.trainer.load(path)
        return epoch
