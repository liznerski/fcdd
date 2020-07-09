from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_confs import default_cifar10_conf


class Cifar10ClassesRunner(ClassesRunner):
    def add_parser_params(self, parser):
        parser = default_cifar10_conf(parser)
        parser.add_argument('--it', type=int, default=5, help='how many times to repeat exp')
        return parser


if __name__ == '__main__':
    runner = Cifar10ClassesRunner()
    runner.args.logdir += '_cifar10_'
    runner.run()
    print()

