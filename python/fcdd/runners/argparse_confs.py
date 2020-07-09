from fcdd.models import choices
from fcdd.datasets import DS_CHOICES, PREPROC_CHOICES
from fcdd.training.setup import OBJECTIVE_PARAMS_DEFAULT, OBJECTIVES, SUPERVISE_MODES, SUPERVISE_PARAMS_DEFAULT
from fcdd.util.argparse import StoreDictKeyValPair
import os.path as pt


def default_conf(parser):
    parser.add_argument('--logdir', type=str, default=pt.join('..', '..', 'data', 'results', 'fcdd_{t}'),
                        help='Directory where logs etc are stored. {t} is replaced by starttime.')
    parser.add_argument('--logdir-suffix', type=str, default='',
                        help='Suffix for directory where logs etc are stored. {t} is replaced by starttime.')
    parser.add_argument('--datadir', type=str, default=pt.join('..', '..', 'data', 'datasets'))
    parser.add_argument('--readme', type=str, default='', help='Some notes to be stored in config.txt')
    parser.add_argument('--optimizer-type', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--scheduler-type', type=str, default='lambda', choices=['lambda', 'milestones'])

    parser.add_argument(
        '--objective-params', action=StoreDictKeyValPair, nargs='+', metavar='KEY=VAL',
        choices=OBJECTIVE_PARAMS_DEFAULT, default=None,
        help="""
        defines specific objective related parameters: 
        heatmaps: {None, grad} with None being the FCDD heatmaps,
        blur_heatmaps: {True, False},
        gaus_std: {None, float} where None uses the formula presented in the paper,
        resdown: {int} the resolution of the logged pictures
        """
    )

    parser.add_argument(
        '--supervise-mode', type=str, default='unsupervised', choices=SUPERVISE_MODES
    )
    parser.add_argument(
        '--supervise-params', action=StoreDictKeyValPair, nargs='+', metavar='KEY=VAL',
        choices=SUPERVISE_PARAMS_DEFAULT, default=None,
        help="""
        defines specific supervise-mode related parameters: 
        noise_mode: {...} see datasets/noise_modes.py and datasets/online_superviser.py,
        online: {True, False},
        limit: {int} amount of OE,
        nominal_label: {0, 1} where 1 swaps labels
        """
    )

    parser.add_argument('--cpu', dest='cuda', action='store_false')
    parser.add_argument('--bias', dest='bias', action='store_true')
    parser.add_argument('--acc-batches', type=int, default=1, help='accumulate batches for SGD for stable gradient')
    parser.add_argument(
        '--viz-ids', type=str, default=None,
        help='directory that contains class-seed-wise experiments, '
             'uses the indices that have been used in the according seeds and classes to additionally generate'
             'heatmap pictures for those samples, stores them in specific_viz_ids subfolder'
    )
    parser.set_defaults(cuda=True, bias=False)
    return parser


def default_fmnist_conf(parser):
    parser = default_conf(parser)
    parser.add_argument('-b', '--batch-size', type=int, default=200)
    parser.add_argument('-e', '--epochs', type=int, default=400)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument('--lr-sched-param', type=float, nargs='*', default=[0.98])
    parser.add_argument('--load', type=str, default=None, help='load snapshot of model')
    parser.add_argument('-d', '--dataset', type=str, default='fmnist', choices=DS_CHOICES)
    parser.add_argument('-n', '--net', type=str, default='SPACEN_CNN28_FCONV', choices=choices())
    parser.add_argument('--final-dim', type=int, default=49, help='needs to be set correctly per net')
    parser.add_argument('--preproc', type=str, default='aeaug1', choices=PREPROC_CHOICES)
    parser.add_argument('--objective', type=str, default='spatial_center', choices=OBJECTIVES)
    parser.add_argument('--quantile', type=float, default=0.93, help='quantile for heatmap pictures normalization')
    return parser


def default_cifar10_conf(parser):
    parser = default_conf(parser)
    parser.add_argument('-b', '--batch-size', type=int, default=200)
    parser.add_argument('-e', '--epochs', type=int, default=600)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument('--lr-sched-param', type=float, nargs='*', default=[0.99])
    parser.add_argument('--load', type=str, default=None, help='load snapshot of model')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=DS_CHOICES)
    parser.add_argument('-n', '--net', type=str, default='SPACEN_CNN32_FCONV_S', choices=choices())
    parser.add_argument('--final-dim', type=int, default=64, help='needs to be set correctly per net')
    parser.add_argument('--preproc', type=str, default='aug1', choices=PREPROC_CHOICES)
    parser.add_argument('--objective', type=str, default='spatial_center', choices=OBJECTIVES)
    parser.add_argument('--quantile', type=float, default=0.93, help='quantile for heatmap pictures normalization')
    return parser


def default_mvtec_conf(parser):
    parser = default_conf(parser)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument('--lr-sched-param', type=float, nargs='*', default=[0.985])
    parser.add_argument('--load', type=str, default=None, help='load snapshot of model')
    parser.add_argument('-d', '--dataset', type=str, default='mvtec', choices=DS_CHOICES)
    parser.add_argument('-n', '--net', type=str, default='SPACEN_CNN224_FCONV', choices=choices())
    parser.add_argument('--final-dim', type=int, default=784, help='needs to be set correctly per net')
    parser.add_argument('--preproc', type=str, default='aeaug1', choices=PREPROC_CHOICES)
    parser.add_argument('--objective', type=str, default='spatial_center', choices=OBJECTIVES)
    parser.add_argument('--raw-shape', type=int, default=240)
    parser.add_argument('--quantile', type=float, default=0.97, help='quantile for heatmap pictures normalization')
    return parser


def default_imagenet_conf(parser):
    parser = default_conf(parser)
    parser.add_argument('-b', '--batch-size', type=int, default=200)
    parser.add_argument('-e', '--epochs', type=int, default=600)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument('--lr-sched-param', type=float, nargs='*', default=[0.99])
    parser.add_argument('--load', type=str, default=None, help='load snapshot of model')
    parser.add_argument('-d', '--dataset', type=str, default='imagenet', choices=DS_CHOICES)
    parser.add_argument('-n', '--net', type=str, default='SPACEN_CNN224_FCONV', choices=choices())
    parser.add_argument('--final-dim', type=int, default=784, help='needs to be set correctly per net')
    parser.add_argument('--preproc', type=str, default='aug1', choices=PREPROC_CHOICES)
    parser.add_argument('--objective', type=str, default='spatial_center', choices=OBJECTIVES)
    parser.add_argument('--quantile', type=float, default=0.97, help='quantile for heatmap pictures normalization')
    return parser


def default_pascalvoc_conf(parser):
    parser = default_conf(parser)
    parser.add_argument('-b', '--batch-size', type=int, default=200)
    parser.add_argument('-e', '--epochs', type=int, default=600)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument('--lr-sched-param', type=float, nargs='*', default=[0.99])
    parser.add_argument('--load', type=str, default=None, help='load snapshot of model')
    parser.add_argument('-d', '--dataset', type=str, default='pascalvoc', choices=DS_CHOICES)
    parser.add_argument('-n', '--net', type=str, default='SPACEN_CNN224_FCONV', choices=choices())
    parser.add_argument('--final-dim', type=int, default=784, help='needs to be set correctly per net')
    parser.add_argument('--preproc', type=str, default='aug1', choices=PREPROC_CHOICES)
    parser.add_argument('--objective', type=str, default='spatial_center', choices=OBJECTIVES)
    parser.add_argument('--quantile', type=float, default=0.97, help='quantile for heatmap pictures normalization')
    return parser

