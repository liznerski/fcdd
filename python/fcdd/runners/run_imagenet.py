from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_confs import default_imagenet_conf


class ImageNetClassesRunner(ClassesRunner):
    def add_parser_params(self, parser):
        parser = default_imagenet_conf(parser)
        parser.add_argument('--it', type=int, default=5, help='how many times to repeat exp')
        return parser


if __name__ == '__main__':
    runner = ImageNetClassesRunner()
    runner.args.logdir += '_imagenet_'
    runner.run()
    runner.copy_files_to_super('WARNINGS.log')
    print()

