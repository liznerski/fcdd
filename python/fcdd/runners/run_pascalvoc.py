from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_confs import default_pascalvoc_conf


class VOCClassesRunner(ClassesRunner):
    def add_parser_params(self, parser):
        parser = default_pascalvoc_conf(parser)
        parser.add_argument('--it', type=int, default=5, help='how many times to repeat exp')
        return parser


if __name__ == '__main__':
    runner = VOCClassesRunner()
    runner.args.logdir += '_pascalvoc_'
    runner.run()
    print()
