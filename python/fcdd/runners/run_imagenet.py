from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_configs import DefaultImagenetConfig


class ImagenetConfig(DefaultImagenetConfig):
    def __call__(self, parser):
        parser = super().__call__(parser)
        parser.add_argument('--it', type=int, default=5, help='Number of runs per class with different random seeds.')
        parser.add_argument(
            '--cls-restrictions', type=int, nargs='+', default=None,
            help='Run only training sessions for some of the classes being nominal.'
        )
        return parser


if __name__ == '__main__':
    runner = ClassesRunner(ImagenetConfig())
    runner.args.logdir += '_imagenet_'
    runner.run()
    print()

