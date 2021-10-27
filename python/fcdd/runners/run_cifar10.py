import torch
from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_configs import DefaultCifar10Config


class Cifar10Config(DefaultCifar10Config):
    def __call__(self, parser):
        parser = super().__call__(parser)
        parser.add_argument('--it', type=int, default=5, help='Number of runs per class with different random seeds.')
        return parser


if __name__ == '__main__':
    torch.set_num_threads(4)
    runner = ClassesRunner(Cifar10Config())
    runner.args.logdir += '_cifar10_'
    runner.run()
    print()

