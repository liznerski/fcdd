from fcdd.datasets.image_folder import extract_custom_classes
from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_configs import DefaultConfig
import fcdd.datasets


class CustomConfig(DefaultConfig):
    def __call__(self, parser):
        parser = super().__call__(parser)
        parser.add_argument('--it', type=int, default=5, help='Number of runs per class with different random seeds.')
        parser.add_argument(
            '--cls-restrictions', type=int, nargs='+', default=None,
            help='Run only training sessions for some of the classes being normal.'
        )
        parser.add_argument(
            '--one-vs-rest', '-ovr', action='store_true',
            help='Activates the one-vs-rest evaluation; i.e., the current class is considered normal during testing '
                 'while all other classes are considered anomalous. This requires the dataset folder to be of '
                 'the form `data/custom/test/classX`, `data/custom/test/classY`, etc., with the class folders '
                 'containing the respective images. Otherwise (i.e., if one-vs-rest is deactivated) the '
                 'dataset folder needs to be of the form `data/custom/test/classX/normal` and '
                 '`data/custom/test/classX/anomalous`, containing the respective normal and anomalous samples '
                 'per class. This also holds for the training set (in case of some known anomalies for a '
                 'semi-supervised setting; set --supervise-mode to `other` to take them into account). '
                 'For more details see :class:`fcdd.datasets.image_folder.ADImageFolderDataset`.'
        )
        parser.add_argument(
            '--ground-truth-maps', '-gtms', action='store_true',
            help='Activates utilization of binary ground-truth maps for training and/or testing. '
                 'This requires additional dataset folders `data/custom/train_maps/classX/...` '
                 'and/or `data/custom/test_maps/classX/...` where the corresponding maps are placed. '
                 'The ground-truth maps need to be binary; i.e., need to be in {0, 255}^{1 x h x w}, '
                 'where 255 marks anomalous regions. The file names of the ground-truth maps need to exactly match '
                 'the file names of the corresponding images, including the file extension.'
                 'For more details see :class:`fcdd.datasets.image_folder_gtms.ADImageFolderDatasetGTM`.'
        )
        parser.set_defaults(one_vs_rest=False, ground_truth_maps=False)
        return parser


if __name__ == '__main__':
    runner = ClassesRunner(CustomConfig())
    runner.args.logdir += '_custom_'
    fcdd.datasets.CUSTOM_CLASSES = extract_custom_classes(runner.args.datadir)
    fcdd.datasets.image_folder.ADImageFolderDataset.ovr = runner.args.one_vs_rest
    fcdd.datasets.image_folder.ADImageFolderDataset.gtm = runner.args.ground_truth_maps
    del runner.args.one_vs_rest
    del runner.args.ground_truth_maps
    runner.run()
    print()

