from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_confs import default_mvtec_conf


class MVTECClassesRunner(ClassesRunner):
    def add_parser_params(self, parser):
        parser = default_mvtec_conf(parser)
        parser.add_argument('--it', type=int, default=5, help='how many times to repeat exp')
        return parser


if __name__ == '__main__':
    runner = MVTECClassesRunner()
    runner.args.logdir += '_mvtec_'
    runner.run()
    print()
