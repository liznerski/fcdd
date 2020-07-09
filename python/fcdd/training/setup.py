import os.path as pt

import torch
import torch.optim as optim
from fcdd.datasets import load_dataset
from fcdd.datasets.bases import GTMapADDataset
from fcdd.models import load_nets
from fcdd.util.logging import ItLogger

OBJECTIVE_PARAMS_DEFAULT = {
    'spatial_center': {
        'heatmaps': None,  # None means using (output-pixel)-wise loss
        'blur_heatmaps': False,
        'gaus_std': None,  # None means using default value
        'resdown': 64
    },
    'hard_boundary': {'heatmaps': 'grad', 'blur_heatmaps': True, 'gaus_std': None, 'resdown': 64},
    'autoencoder': {'heatmaps': None, 'blur_heatmaps': True, 'gaus_std': None, 'resdown': 64},
}
OBJECTIVES = tuple(OBJECTIVE_PARAMS_DEFAULT.keys())

SUPERVISE_PARAMS_DEFAULT = {
    'unsupervised': {'online': False},
    'other': {'online': False},
    'noise': {'noise_mode': 'gaussian', 'online': True, 'limit': 10000000000000, 'nominal_label': 0},
    'malformed_normal': {'noise_mode': 'gaussian', 'online': True, 'nominal_label': 0},
    'noise_gt': {'noise_mode': 'gaussian', 'online': True, 'nominal_label': 0},
    'malformed_normal_gt': {'noise_mode': 'gaussian', 'online': True, 'nominal_label': 0},
}
SUPERVISE_MODES = tuple(SUPERVISE_PARAMS_DEFAULT.keys())


def pick_opt_sched(net, lr, wdk, sched_params, opt, sched):
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
        cuda: bool, dataset: str, datapath: str, normal_class: int, net: str, final_dim: int,
        lr: float, wdk: float, lr_schedule: [float], logdir: str, config: str,
        batch_size: int = 128, workers: int = 1, objective: str = 'hard_boundary',
        log_start_time: int = None, objective_params: dict = None,
        preproc: str = 'ae', bias: bool = False,
        supervise_mode: str = 'unsupervised', supervise_params: dict = None, raw_shape: int = 240,
        quantile: float = 0.93, optimizer_type: str = 'sgd', scheduler_type: str = 'adam'
):
    assert objective in OBJECTIVES
    objective_params_updated = dict(OBJECTIVE_PARAMS_DEFAULT[objective])
    objective_params_updated.update(objective_params or {})
    assert supervise_mode in SUPERVISE_MODES
    supervise_params_updated = dict(SUPERVISE_PARAMS_DEFAULT[supervise_mode])
    supervise_params_updated.update(supervise_params or {})
    device = torch.device('cuda:0') if cuda else torch.device('cpu')
    logger = ItLogger(pt.abspath(pt.join(logdir, '')), exp_start_time=log_start_time)
    ds = load_dataset(
        dataset, pt.abspath(pt.join(datapath, '')), normal_class, preproc, supervise_mode, supervise_params_updated,
        raw_shape, logger=logger
    )
    loaders = ds.loaders(batch_size=batch_size, num_workers=workers)
    net, pt_net = load_nets(net, final_dim, ds.shape, bias=bias, **objective_params_updated)
    net = pt_net if objective == 'autoencoder' else net
    logger.logtxt(
        '##### NET RECEPTION {} #####'.format(net.reception if hasattr(net, 'reception') else None), print=True
    )
    net = net.to(device)
    optimizer, scheduler = pick_opt_sched(net, lr, wdk, lr_schedule, optimizer_type, scheduler_type)
    logger.save_params(net, config)
    if not hasattr(ds, 'nominal_label') or ds.nominal_label < ds.anomalous_label:
        ds_order = ['norm', 'anom']
    else:
        ds_order = ['anom', 'norm']
    logger.imsave(
        'ds_preview', ds.preview(20), nrow=20,
        rowheaders=ds_order if not isinstance(ds.train_set, GTMapADDataset)
        else [*ds_order, '', *['gtno' if s == 'norm' else 'gtan' for s in ds_order]]
    )
    return {
        'net': net, 'dataset_loaders': loaders, 'opt': optimizer, 'sched': scheduler, 'logger': logger,
        'device': device, 'objective': objective,
        'objective_params': objective_params_updated,
        'supervise_params': supervise_params_updated,
        'quantile': quantile
    }


