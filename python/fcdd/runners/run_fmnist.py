import torch
from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_configs import DefaultFmnistConfig


class FmnistConfig(DefaultFmnistConfig):
    def __call__(self, parser):
        parser = super().__call__(parser)
        parser.add_argument('--it', type=int, default=5, help='Number of runs per class with different random seeds.')
        return parser


if __name__ == '__main__':
    torch.set_num_threads(4)
    runner = ClassesRunner(FmnistConfig())
    runner.args.logdir += '_fmnist_'
    runner.run()
    print()

