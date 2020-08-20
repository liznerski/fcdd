"""
This script can be used to rerun an experiment with random seeds or to run the explanation baseline experiments.
It loads the parameter configuration for a given experiment (we name this one the source experiment)
and runs another set of experiments with the same base configuration.
Additional to the usual log data,
it produces heatmap images for the same input images that have been logged in the source experiment.
These can be found in logdir/specific_viz_ids. The script stores its log data at the same path
as the source experiment, but adds a suffix, like "_rerun", "_hsc", or "_ae".
For instance, if the source experiment path is data/my/exp then the rerun would store its data at data/my/exp_rerun.
"""

import os
import os.path as pt
from fcdd.runners.bases import ClassesRunner
from fcdd.runners.argparse_configs import DefaultConfig
from fcdd.util.io import combine_specific_viz_ids_pics, OPTIONS, extract_args, read_cfg


class ExpConfig(DefaultConfig):
    def __call__(self, parser):
        parser.add_argument(
            'srcexp', type=str, default=None,
            help='Root directory for log data of some FCDD experiment, '
                 'loads basic configuration and produces additional heatmap images for logged input images.'
        )
        parser.add_argument(
            '--exps', type=str, nargs='+', default=['hsc', 'ae'], choices=OPTIONS,
            help='Types of experiments to run, multiple inputs result in experiments run sequentially. '
                 '"base" reruns the experiment, others transform parts of the configuration.'
                 '"hsc" adds fully connected layer to the net and produces heatmaps by computing the gradients.'
                 '"ae" transforms the net to an hourglass architecture, trains an autoencoder, and uses the '
                 'pixel-wise reconstruction loss to produce heatmaps.'
        )
        parser.add_argument(
            '--its-restrictions', type=int, nargs='+', default=None,
            help='run only a part of the seeds found in the source experiment'
        )
        parser.add_argument(
            '--cls-restrictions', nargs='+', default=None, type=int,
            help='run only a part of the classes found in the source experiment'
        )
        return parser


if __name__ == '__main__':
    runner = ClassesRunner(ExpConfig())
    assert runner.args.srcexp is not None, 'directory of source experiment is not given!'

    # find all class folders, where each folder contains log data for a different class taken as nominal
    cls_dirs = [
        pt.join(runner.args.srcexp, d) for d in os.listdir(runner.args.srcexp)
        if pt.isdir(pt.join(runner.args.srcexp, d))
    ]
    cls_dirs = [c for c in sorted(cls_dirs, key=os.path.getmtime) if c.split(os.sep)[-1].startswith('normal_')]
    assert len(cls_dirs) > 0, 'source experiment directory has no class folders, i.e. contains no log data!'

    # find all seed folders, where each folder contains log data for a different random seed
    it_dirs = [
        pt.join(cls_dirs[0], d) for d in os.listdir(cls_dirs[0])
        if pt.isdir(pt.join(cls_dirs[0], d))
    ]
    it_dirs = [i for i in sorted(it_dirs, key=os.path.getmtime) if i.split(os.sep)[-1].startswith('it_')]
    assert len(it_dirs) > 0, "source experiment directory's first class folder has no runs, i.e. contains no log data!"

    # read configuration file, where the choice of parameters of the source experiment is logged
    assert 'config.txt' in os.listdir(pt.join(cls_dirs[0], it_dirs[0])), 'could not find configuration file!'
    cfgfile = pt.join(cls_dirs[0], it_dirs[0], 'config.txt')
    cfg = read_cfg(cfgfile)
    extract_args(runner.args, cfg)

    # start experiments sequentially (order fixed)
    dirs = []
    runner.args.it = len(it_dirs)
    vizids_dir = runner.args.srcexp
    exps = runner.args.exps
    del runner.args.srcexp
    del runner.args.normal_class
    del runner.args.exps
    if 'base' in exps:
        runner.args.logdir = vizids_dir + "_rerun"
        runner.run()
        dirs.append(runner.get_base_logdir())
    else:
        runner.args.logdir = vizids_dir
        dirs.append(vizids_dir)
        exps.insert(0, 'base')
    if 'hsc' in exps:
        runner.arg_to_hsc()
        runner.run()
        dirs.append(runner.get_base_logdir())
    if 'ae' in exps:
        runner.arg_to_ae()
        runner.run()
        dirs.append(runner.get_base_logdir())

    # combine the heatmaps for images that all experiments have in common, resulting in images like in the paper
    combine_specific_viz_ids_pics(dirs, setup=exps, skip_further=True)
    print()
