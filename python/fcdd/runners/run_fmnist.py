from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_confs import default_fmnist_conf


class FMNISTClassesRunner(ClassesRunner):
    def add_parser_params(self, parser):
        parser = default_fmnist_conf(parser)
        parser.add_argument('--it', type=int, default=5, help='how many times to repeat exp')
        return parser


if __name__ == '__main__':
    runner = FMNISTClassesRunner()
    runner.args.logdir += '_fmnist_'
    runner.run()
    print()


