"""
Script to load the experiment configs from a given experiment.
Then runs another set of experiments with the same base configs and produces heatmap pictures for the same indices.
These pictures can be found in logdir/specific_viz_ids.
With this script the baseline experiments can be run (ae and hsc),
but it can also be used to rerun the experiment per se.
"""

import os
import os.path as pt
from fcdd.runners.bases import ClassesRunner
from fcdd.util.io import combine_specific_viz_ids_pics, OPTIONS, extract_args, read_cfg


class ExpRunner(ClassesRunner):
    def argparse(self, f_trans_args=None):
        from fcdd.util.argparse import ArgumentParser
        parser = ArgumentParser(description=__doc__)

        parser = self.add_parser_params(parser)

        if f_trans_args is not None:
            f_trans_args(parser)

        args = parser.parse_args()
        return args

    def add_parser_params(self, parser):
        parser.add_argument(
            'vizids', type=str, default=None,
            help='dictionary that contains class-seed-wise experiments, '
                 'uses the indices that have been used in the according its and classes viz here too (additionally)'
        )
        parser.add_argument(
            '--exps', type=str, nargs='+', default=['hsc', 'ae'], choices=OPTIONS,
            help='type of experiment, "base" reruns experiment, others transform parts of the config to match'
                 'the new experiment. For instance "hsc" adds fully connected layer to the net and'
                 'runs the hsc objective with gradient heatmaps, but shares the rest of the config. '
        )
        parser.add_argument(
            '--restrict-its', type=int, nargs='+', default=None, help='run only a part of the seeds'
        )
        parser.add_argument(
            '--restrict-cls', nargs='+', default=None, type=int, help='run only a part of the classes'
        )
        return parser


if __name__ == '__main__':
    runner = ExpRunner()
    assert runner.args.vizids is not None and 'base' not in runner.args.exps
    cls_dirs = [
        pt.join(runner.args.vizids, d) for d in os.listdir(runner.args.vizids)
        if pt.isdir(pt.join(runner.args.vizids, d))
    ]
    cls_dirs = sorted(cls_dirs, key=os.path.getmtime)
    it_dirs = [
        pt.join(cls_dirs[0], d) for d in os.listdir(cls_dirs[0])
        if pt.isdir(pt.join(cls_dirs[0], d))
    ]
    it_dirs = sorted(it_dirs, key=os.path.getmtime)
    cfgfile = pt.join(cls_dirs[0], it_dirs[0], 'config.txt')
    cfg = read_cfg(cfgfile)
    extract_args(runner.args, cfg)
    dirs = []
    runner.args.cls_restrictions = runner.args.restrict_cls
    del runner.args.restrict_cls
    runner.args.it = len(it_dirs)
    del runner.args.normal_class
    vizids_dir = runner.args.vizids
    del runner.args.vizids
    exps = runner.args.exps
    del runner.args.exps
    if 'base' in exps:
        runner.args.logdir = vizids_dir + "_rerun"
        runner.run()
        runner.copy_files_to_super('WARNINGS.log')
        dirs.append(runner.get_base_logdir())
    else:
        runner.args.logdir = vizids_dir
        dirs.append(vizids_dir)
        exps.insert(0, 'base')
    if 'hsc' in exps:
        runner.arg_to_hsc()
        runner.run()
        runner.copy_files_to_super('WARNINGS.log')
        dirs.append(runner.get_base_logdir())
    if 'ae' in exps:
        runner.arg_to_ae()
        runner.run()
        runner.copy_files_to_super('WARNINGS.log')
        dirs.append(runner.get_base_logdir())

    combine_specific_viz_ids_pics(dirs, setup=exps, skip_further=True)
    print()
