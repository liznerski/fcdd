import os.path as pt
from argparse import ArgumentParser

import numpy as np
from fcdd.datasets import DS_CHOICES, PREPROC_CHOICES
from fcdd.datasets.noise_modes import MODES
from fcdd.models import choices
from fcdd.training.setup import OBJECTIVES, SUPERVISE_MODES


class DefaultConfig(object):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        """
        Defines all the arguments for running an FCDD experiment.
        :param parser: instance of an ArgumentParser.
        :return: the parser with added arguments
        """

        # define directories for datasets and logging
        parser.add_argument(
            '--logdir', type=str, default=pt.join('..', '..', 'data', 'results', 'fcdd_{t}'),
            help='Directory where log data is to be stored. The pattern {t} is replaced by the start time. '
                 'Defaults to ../../data/results/fcdd_{t}. '
        )
        parser.add_argument(
            '--logdir-suffix', type=str, default='',
            help='String suffix for log directory, again {t} is replaced by the start time. '
        )
        parser.add_argument(
            '--datadir', type=str, default=pt.join('..', '..', 'data', 'datasets'),
            help='Directory where datasets are found or to be downloaded to. Defaults to ../../data/datasets.',
        )
        parser.add_argument(
            '--viz-ids', type=str, default=None,
            help='Directory that contains log data of an old experiment. '
                 'When given, in addition to the usual log data and heatmaps, the training produces heatmaps for the '
                 'same images that have been logged in the according seeds and class runs found in the directory.'
        )
        parser.add_argument(
            '--readme', type=str, default='',
            help='Some notes to be stored in the automatically created config.txt configuration file.'
        )

        # training parameters
        parser.add_argument(
            '--objective', type=str, default='fcdd', choices=OBJECTIVES,
            help='Chooses the objective to run explanation baseline experiments. Defaults to FCDD.'
        )
        parser.add_argument('-b', '--batch-size', type=int, default=128)
        parser.add_argument('-e', '--epochs', type=int, default=200)
        parser.add_argument('-w', '--workers', type=int, default=4)
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
        parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
        parser.add_argument(
            '--optimizer-type', type=str, default='sgd', choices=['sgd', 'adam'],
            help='The type of optimizer. Defaults to "sgd". '
        )
        parser.add_argument(
            '--scheduler-type', type=str, default='lambda', choices=['lambda', 'milestones'],
            help='The type of learning rate scheduler. Either "lambda", which reduces the learning rate each epoch '
                 'by a certain factor, or "milestones", which sets the learning rate to certain values at certain '
                 'epochs. Defaults to "lambda"'
        )
        parser.add_argument(
            '--lr-sched-param', type=float, nargs='*', default=[0.985],
            help='Sequence of learning rate scheduler parameters. '
                 'For the "lambda" scheduler, just one parameter is allowed, '
                 'which sets the factor the learning rate is reduced per epoch. '
                 'For the "milestones" scheduler, at least two parameters are needed, '
                 'the first determining the factor by which the learning rate is reduced at each milestone, '
                 'and the others being each a milestone. For instance, "0.1 100 200 300" reduces the learning rate '
                 'by 0.1 at epoch 100, 200, and 300. '
        )
        parser.add_argument(
            '--load', type=str, default=None,
            help='Path to a file that contains a snapshot of the network model. '
                 'When given, the network loads the found weights and state of the training. '
                 'If epochs are left to be trained, the training is continued. '
                 'Note that only one snapshot is given, thus using a runner that trains for multiple different classes '
                 'to be nominal is not applicable. '
        )
        parser.add_argument('-d', '--dataset', type=str, default='custom', choices=DS_CHOICES)
        parser.add_argument(
            '-n', '--net', type=str, default='FCDD_CNN224_VGG_F', choices=choices(),
            help='Chooses a network architecture to train. Note that not all architectures fit every objective. '
        )
        parser.add_argument(
            '--preproc', type=str, default='aug1', choices=PREPROC_CHOICES,
            help='Determines the kind of preprocessing pipeline (augmentations and such). '
                 'Have a look at the code (dataset implementation, e.g. fcdd.datasets.cifar.py) for details.'
        )
        parser.add_argument(
            '--acc-batches', type=int, default=1,
            help='To speed up data loading, '
                 'this determines the number of batches that are accumulated to be used for training. '
                 'For instance, acc_batches=2 iterates the data loader two times, concatenates the batches, and '
                 'passes the result to the further training procedure. This has no impact on the performance '
                 'if the batch size is reduced accordingly (e.g. one half in this example), '
                 'but can decrease training time. '
        )
        parser.add_argument('--no-bias', dest='bias', action='store_false', help='Uses no bias in network layers.')
        parser.add_argument('--cpu', dest='cuda', action='store_false', help='Trains on CPU only.')

        # artificial anomaly settings
        parser.add_argument(
            '--supervise-mode', type=str, default='noise', choices=SUPERVISE_MODES,
            help='This determines the kind of artificial anomalies. '
                 '"unsupervised" uses no anomalies at all. '
                 '"other" uses ground-truth anomalies. '
                 '"noise" uses pure noise images or Outlier Exposure. '
                 '"malformed_normal" adds noise to nominal images to create malformed nominal anomalies. '
                 '"malformed_normal_gt" is like malformed_normal, but with ground-truth anomaly heatmaps for training. '
        )
        parser.add_argument(
            '--noise-mode', type=str, default='imagenet22k', choices=MODES,
            help='The type of noise used when artificial anomalies are activated. Dataset names refer to OE. '
                 'See fcdd.datasets.noise_modes.py.'
        )
        parser.add_argument(
            '--oe-limit', type=int, default=np.infty,
            help='Determines the amount of different samples used for Outlier Exposure. '
                 'Has no impact on synthetic anomalies.'
        )
        parser.add_argument(
            '--offline-supervision', dest='online_supervision', action='store_false',
            help='Instead of sampling artificial anomalies during training by having a 50%% chance to '
                 'replace nominal samples, this mode samples them once at the start of the training and adds them to '
                 'the training set. '
                 'This yields less performance and higher RAM utilization, but reduces the training time. '
        )
        parser.add_argument(
            '--nominal-label', type=int, default=0,
            help='Determines the label that marks nominal samples. '
                 'Note that this is not the class that is considered nominal! '
                 'For instance, class 5 is the nominal class, which is labeled with the nominal label 0.'
        )

        # heatmap generation parameters
        parser.add_argument(
            '--blur-heatmaps', dest='blur_heatmaps', action='store_true',
            help='Blurs heatmaps, like done for the explanation baseline experiments in the paper.'
        )
        parser.add_argument(
            '--gauss-std', type=float, default=10,
            help='Sets a constant value for the standard deviation of the Gaussian kernel used for upsampling and '
                 'blurring.'
        )
        parser.add_argument(
            '--quantile', type=float, default=0.97,
            help='The quantile that is used to normalize the generated heatmap images. '
                 'This is explained in the Appendix of the paper.'
        )
        parser.add_argument(
            '--resdown', type=int, default=64,
            help='Sets the maximum resolution of logged images (per heatmap), images will be downsampled '
                 'if they exceed this threshold. For instance, resdown=64 makes every image of heatmaps contain '
                 'individual heatmaps and inputs of width 64 and height 64 at most.'
        )

        parser.set_defaults(cuda=True, bias=True, blur_heatmaps=False, online_supervision=True)
        return parser


class DefaultFmnistConfig(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.set_defaults(
            batch_size=128, epochs=400, learning_rate=1e-2,
            weight_decay=1e-6, lr_sched_param=[0.98], dataset='fmnist',
            net='FCDD_CNN28_W', quantile=0.85, noise_mode='cifar100',
            preproc='lcnaug1', gauss_std=1.2,
        )
        return parser


class DefaultCifar10Config(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.set_defaults(
            batch_size=20, acc_batches=10, epochs=600,
            optimizer_type='adam', scheduler_type='milestones',
            lr_sched_param=[0.1, 400, 500], dataset='cifar10',
            net='FCDD_CNN32_LW3K', quantile=0.85,
            noise_mode='cifar100', gauss_std=1.2,
        )
        return parser


class DefaultMvtecConfig(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.set_defaults(
            batch_size=16, acc_batches=8, supervise_mode='malformed_normal',
            gauss_std=12, weight_decay=1e-4, epochs=200, preproc='lcnaug1',
            quantile=0.99, net='FCDD_CNN224_VGG_F', dataset='mvtec', noise_mode='confetti'
        )
        return parser


class DefaultImagenetConfig(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.set_defaults(
            batch_size=20, acc_batches=10, epochs=600,
            optimizer_type='adam', scheduler_type='milestones',
            lr_sched_param=[0.1, 400, 500], noise_mode='imagenet22k',
            dataset='imagenet', gauss_std=8, net='FCDD_CNN224_VGG_NOPT'
        )
        return parser


class DefaultPascalvocConfig(DefaultConfig):
    def __call__(self, parser: ArgumentParser):
        parser = super().__call__(parser)
        parser.set_defaults(
            batch_size=20, acc_batches=10, epochs=600,
            optimizer_type='adam', scheduler_type='milestones', lr_sched_param=[0.1, 400, 500],
            dataset='pascalvoc', noise_mode='imagenet', net='FCDD_CNN224_VGG_NOPT',
            nominal_label=1, gauss_std=8, quantile=0.99,
        )
        return parser

