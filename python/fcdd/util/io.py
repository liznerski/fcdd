import json
import re
import os
import os.path as pt
import torch
import numpy as np
import warnings
import sys
from .logging import Logger


def read_cfg(cfg_file):
    with open(cfg_file) as reader:
        cfg = reader.readlines()
        cfg = ' '.join(cfg)
        re.DOTALL = True
        pttn = re.compile('\{(.*)\}')
        cfg = pttn.findall(cfg)
        assert len(cfg) == 1
        cfg = cfg[0]
        cfg = json.loads('{{{}}}'.format(cfg))
        return cfg


def extract_args(args, cfg):
    args.bias = cfg['kwargs']['bias']
    args.objective_params = cfg['kwargs']['objective_params']
    args.optimizer_type = cfg['kwargs']['optimizer_type']
    args.preproc = cfg['kwargs']['preproc']
    args.quantile = cfg['kwargs']['quantile']
    args.scheduler_type = cfg['kwargs']['scheduler_type']
    args.supervise_mode = cfg['kwargs']['supervise_mode']
    args.supervise_params = cfg['kwargs']['supervise_params']
    args.batch_size = cfg['batch_size']
    args.epochs = cfg['epochs']
    args.workers = cfg['workers']
    args.final_dim = cfg['final_dim']
    args.learning_rate = cfg['learning_rate']
    args.weight_decay = cfg['weight_decay']
    args.lr_sched_param = cfg['lr_sched_param']
    args.dataset = cfg['dataset']
    args.net = cfg['net']
    args.readme = ''
    args.datadir = cfg['datadir']
    args.normal_class = cfg['normal_class']
    args.acc_batches = cfg['acc_batches']
    args.cuda = True
    args.objective = cfg['objective']
    args.logdir = cfg['logdir']
    args.load = cfg['load']
    return args


OPTIONS = ['base', 'ae', 'hsc', 'gts']


def combine_specific_viz_ids_pics(srcs, out=None, debug=False, setup=('base', 'hsc', 'ae'),
                                  skip_further=False, only_cls=None):
    assert all([s in OPTIONS for s in setup])
    assert setup[0] == 'base'
    if 'gts' in setup:
        assert setup[-1] == 'gts'

    if out is None:
        out = srcs[0] + '_COMBINED_PAPER_PICS'

    if len(srcs) != len(setup):
        raise ValueError(
            'fixed len of src required, {}, but found {}!'
            .format(' '.join(['({}) {}'.format(i + 1, s) for i, s in enumerate(setup)]), len(srcs))
        )
    pics = {}
    for n, src in enumerate(srcs):
        cls_labels = [pt.join(src, c) for c in os.listdir(src)]
        cls_labels.sort(key=pt.getmtime)
        cls_labels = [pt.basename(c) for c in cls_labels]
        if all([c.startswith('it_') for c in cls_labels if pt.isdir(pt.join(src, c))]):  # one class experiment
            cls_labels = ['.']
        for cls_dir in cls_labels:
            if not pt.isdir(pt.join(src, cls_dir)):
                continue
            assert cls_dir.startswith('normal_')
            if only_cls is not None and len(only_cls) > 0 and int(cls_dir[7:]) not in only_cls:
                continue
            print('collecting pictures of {} {}...'.format(src, cls_dir))
            for it_dir in os.listdir(pt.join(src, cls_dir)):
                if pt.isfile(pt.join(src, cls_dir, it_dir)):
                    continue
                cfg = read_cfg(pt.join(src, cls_dir, it_dir, 'config.txt'))
                tims_dir = pt.join(src, cls_dir, it_dir, 'tims')
                if n == 0:
                    if pt.exists(pt.join(tims_dir, 'specific_viz_ids')):
                        raise ValueError(
                            'First src should not contains specific viz ids, as first src should be the base!')
                    for root, dirs, files in os.walk(tims_dir):
                        for f in files:
                            assert f[-4:] == '.pth'
                            if cls_dir not in pics:
                                pics[cls_dir] = {}
                            if it_dir not in pics[cls_dir]:
                                pics[cls_dir][it_dir] = {}
                            pics[cls_dir][it_dir][f[:-4]] = [torch.load(pt.join(root, f))]
                else:
                    if not pt.exists(pt.join(tims_dir, 'specific_viz_ids')):
                        raise ValueError('Src {} should contain specific viz ids, but it doesnt!'.format(src))
                    for root, dirs, files in os.walk(pt.join(tims_dir, 'specific_viz_ids')):
                        for f in files:
                            assert f[-4:] == '.pth'
                            if cls_dir == '.' and cls_dir not in pics:
                                warnings.warn('Seems that src {} is a one class experiment...'.format(src))
                                cls = 'normal_{}'.format(cfg['normal_class'])
                            else:
                                cls = cls_dir
                            if cls not in pics or it_dir not in pics[cls]:
                                raise ValueError('{} {} is missing in base src!!'.format(cls_dir, it_dir))
                            if setup[n] in ('ae', ):
                                if not f.startswith('ae_'):
                                    continue
                                pics[cls][it_dir][f[3:-4]].append(torch.load(pt.join(root, f)))
                            else:
                                if f.startswith('ae_'):
                                    raise ValueError(
                                        'ae has been found in position {}, but shouldnt be!'.format(n)
                                    )
                                pics[cls][it_dir][f[:-4]].append(torch.load(pt.join(root, f)))

    logger = Logger(out)

    for cls_dir in pics:
        print('creating pictures for {} {}...'.format(out, cls_dir))
        for it_dir in pics[cls_dir]:
            for file in pics[cls_dir][it_dir]:
                combined_pic = []
                inps = []
                gts = None
                tensors = pics[cls_dir][it_dir][file]
                if len(tensors) != len(srcs):
                    print(
                        'Some specific viz id tims are missing for {} {}!! Skipping them...'.format(cls_dir, it_dir),
                        file=sys.stderr
                    )
                    continue

                # 0 == base src
                t = tensors[0]
                rows, cols, c, h, w = t.shape
                inps.append(t[0])
                if 'gts' in setup:
                    combined_pic.extend([*t[:2 if skip_further else -1]])
                    gts = t[-1]
                else:
                    combined_pic.extend([*t[:2 if skip_further else 10000000000]])

                for t in tensors[1:]:
                    rows, cols, c, h, w = t.shape
                    if rows == 3:  # assume gts in final row
                        t = t[:-1]
                    inps.append(t[0])
                    if debug:
                        combined_pic.extend([*t])
                    else:
                        combined_pic.append(t[1])

                # ADD GTMAP
                if gts is not None:
                    combined_pic.append(gts)

                # check of all inputs have been the same
                for i, s in enumerate(srcs):
                    for j, ss in enumerate(srcs):
                        if j <= i:
                            continue
                        if (inps[i] != inps[j]).sum() > 0:
                            raise ValueError('SRC {} and SRC {} have different inputs!!!'.format(srcs[i], srcs[j]))

                # combine
                new_cols = combined_pic[0].size(0)
                tim = torch.cat(combined_pic)
                logger.imsave(file, tim, nrow=new_cols, scale_mode='none', suffix=pt.join(cls_dir, it_dir))

    print('Successfully combined pics in {}.'.format(out))
