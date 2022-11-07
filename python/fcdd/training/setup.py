import os.path as pt
from typing import List

import torch
import torch.optim as optim
from fcdd.datasets import load_dataset
from fcdd.datasets.bases import GTMapADDataset
from fcdd.datasets.noise_modes import MODES
from fcdd.models import load_nets
from fcdd.models.bases import BaseNet
from fcdd.util.logging import Logger

OBJECTIVES = ('fcdd', 'hsc', 'ae')
SUPERVISE_MODES = ('unsupervised', 'other', 'noise', 'malformed_normal', 'malformed_normal_gt')


def pick_opt_sched(net: BaseNet, lr: float, wdk: float, sched_params: List[float], opt: str, sched: str):
    """
    Creates an optimizer and learning rate scheduler based on the given parameters.
    :param net: some neural network.
    :param lr: initial learning rate.
    :param wdk: weight decay (L2 penalty) regularizer.
    :param sched_params: learning rate scheduler parameters. Format depends on the scheduler type.
        For 'milestones' needs to have at least two elements, the first corresponding to the factor
        the learning rate is decreased by at each milestone, the rest corresponding to milestones (epochs).
        For 'lambda' needs to have exactly one element, i.e. the factor the learning rate is decreased by
        at each epoch.
    :param opt: optimizer type, needs to be one of {'sgd', 'adam'}.
    :param sched: learning rate scheduler type, needs to be one of {'lambda', 'milestones'}.
    :return:
    """
    if net is None:
        return None, None
    if opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=wdk, momentum=0.9, nesterov=True)
    elif opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wdk)
    else:
        raise NotImplementedError('Optimizer type {} not known.'.format(opt))
    if sched == 'lambda':
        assert len(sched_params) == 1 and 0 < sched_params[0] <= 1
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: sched_params[0] ** ep)
    elif sched == 'milestones':
        assert len(sched_params) >= 2 and 0 < sched_params[0] <= 1 and all([p > 1 for p in sched_params[1:]])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(s) for s in sched_params[1:]], sched_params[0], )
    else:
        raise NotImplementedError('LR scheduler type {} not known.'.format(sched))
    return optimizer, scheduler


def trainer_setup(
        dataset: str, datadir: str, logdir: str, net: str, bias: bool,
        learning_rate: float, weight_decay: float, lr_sched_param: List[float], batch_size: int,
        optimizer_type: str, scheduler_type: str,
        objective: str, preproc: str, supervise_mode: str, nominal_label: int,
        online_supervision: bool, oe_limit: int, noise_mode: str,
        workers: int, quantile: float, resdown: int, gauss_std: float, blur_heatmaps: bool,
        cuda: bool, config: str, log_start_time: int = None, normal_class: int = 0,
) -> dict:
    """
    Creates a complete setup for training, given all necessary parameter from a runner (seefcdd.runners.bases.py).
    This includes loading networks, datasets, data loaders, optimizers, and learning rate schedulers.
    :param dataset: dataset identifier string (see :data:`fcdd.datasets.DS_CHOICES`).
    :param datadir: directory where the datasets are found or to be downloaded to.
    :param logdir: directory where log data is to be stored.
    :param net: network model identifier string (see :func:`fcdd.models.choices`).
    :param bias: whether to use bias in the network layers.
    :param learning_rate: initial learning rate.
    :param weight_decay: weight decay (L2 penalty) regularizer.
    :param lr_sched_param: learning rate scheduler parameters. Format depends on the scheduler type.
        For 'milestones' needs to have at least two elements, the first corresponding to the factor
        the learning rate is decreased by at each milestone, the rest corresponding to milestones (epochs).
        For 'lambda' needs to have exactly one element, i.e. the factor the learning rate is decreased by
        at each epoch.
    :param batch_size: batch size, i.e. number of data samples that are returned per iteration of the data loader.
    :param optimizer_type: optimizer type, needs to be one of {'sgd', 'adam'}.
    :param scheduler_type: learning rate scheduler type, needs to be one of {'lambda', 'milestones'}.
    :param objective: the training objective. See :data:`OBJECTIVES`.
    :param preproc: data preprocessing pipeline identifier string (see :data:`fcdd.datasets.PREPROC_CHOICES`).
    :param supervise_mode: the type of generated artificial anomalies.
        See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
    :param nominal_label: the label that is to be returned to mark nominal samples.
    :param online_supervision: whether to sample anomalies online in each epoch,
        or offline before training (same for all epochs in this case).
    :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode).
    :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
    :param workers: how many subprocesses to use for data loading.
    :param quantile: the quantile that is used to normalize the generated heatmap images.
    :param resdown: the maximum resolution of logged images, images will be downsampled if necessary.
    :param gauss_std: a constant value for the standard deviation of the Gaussian kernel used for upsampling and
        blurring, the default value is determined by :func:`fcdd.datasets.noise.kernel_size_to_std`.
    :param blur_heatmaps: whether to blur heatmaps.
    :param cuda: whether to use GPU.
    :param config: some config text that is to be stored in the config.txt file.
    :param log_start_time: the start time of the experiment.
    :param normal_class: the class that is to be considered nominal.
    :return: a dictionary containing all necessary parameters to be passed to a Trainer instance.
    """
    assert objective in OBJECTIVES, 'unknown objective: {}'.format(objective)
    assert supervise_mode in SUPERVISE_MODES, 'unknown supervise mode: {}'.format(supervise_mode)
    assert noise_mode in MODES, 'unknown noise mode: {}'.format(noise_mode)
    device = torch.device('cuda:0') if cuda else torch.device('cpu')
    logger = Logger(pt.abspath(pt.join(logdir, '')), exp_start_time=log_start_time)
    ds = load_dataset(
        dataset, pt.abspath(pt.join(datadir, '')), normal_class, preproc, supervise_mode,
        noise_mode, online_supervision, nominal_label, oe_limit, logger=logger
    )
    loaders = ds.loaders(batch_size=batch_size, num_workers=workers)
    net = load_nets(net, ds.shape, bias=bias)
    logger.logtxt(
        '##### NET RECEPTION {} #####'.format(net.reception if hasattr(net, 'reception') else None), print=True
    )
    net = net.to(device)
    optimizer, scheduler = pick_opt_sched(
        net, learning_rate, weight_decay, lr_sched_param, optimizer_type, scheduler_type
    )
    logger.save_params(net, config)
    if not hasattr(ds, 'nominal_label') or ds.nominal_label < ds.anomalous_label:
        ds_order = ['norm', 'anom']
    else:
        ds_order = ['anom', 'norm']
    images = ds.preview(20, classes=[0, 1] if supervise_mode != "unsupervised" else [0], train=True)
    logger.imsave(
        'ds_preview', torch.cat([*images]), nrow=images.size(1),
        rowheaders=ds_order if not isinstance(ds.train_set, GTMapADDataset)
        else [*ds_order, '', *['gtno' if s == 'norm' else 'gtan' for s in ds_order]]
    )
    return {
        'net': net, 'dataset_loaders': loaders, 'opt': optimizer, 'sched': scheduler, 'logger': logger,
        'device': device, 'objective': objective, 'quantile': quantile, 'resdown': resdown,
        'gauss_std': gauss_std, 'blur_heatmaps': blur_heatmaps
    }


