from fcdd.runners.bases import SeedsRunner
from fcdd.runners.argparse_configs import DefaultMvtecConfig


class SingleMvtecConfig(DefaultMvtecConfig):
    def __call__(self, parser):
        parser = super().__call__(parser)
        parser.add_argument('--it', type=int, default=1, help='Number of runs with different random seeds.')
        parser.add_argument('--normal-class', type=int, default=0, help='Class that is trained to be nominal.')
        return parser


if __name__ == '__main__':
    runner = SeedsRunner(SingleMvtecConfig())
    runner.args.logdir += '_mvtec_single_run_cls{}_'.format(runner.args.normal_class)
    runner.run()
    print()

