import json
import os.path as pt
import re
import time
import traceback
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy

from fcdd.datasets import no_classes, str_labels
from fcdd.training.setup import trainer_setup
from fcdd.training.super_trainer import ADTrainer
from fcdd.util.logging import plot_many_roc, time_format
from fcdd.util.metrics import mean_roc


def extract_viz_ids(dir, cls, it):
    if dir is None:
        return ()
    logfile = pt.join(dir, 'normal_{}'.format(cls), 'it_{}'.format(it), 'log.txt')
    if not pt.exists(logfile):
        raise ValueError('Trying to extract viz_ids from {}, but file doesnt exist'.format(logfile))
    viz_ids = []
    curlbl = -1
    with open(logfile, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            if line.startswith('Interpretation visualization paper image'):
                labels = re.findall('label .*:', line)
                ids = re.findall(r'\[.*\]', line)
                assert len(labels) == 1 and len(ids) == 1
                assert int(labels[0][6:-1]) > curlbl and int(labels[0][6:-1]) in [0, 1]
                curlbl = int(labels[0][6:-1])
                viz_ids.append(json.loads(re.findall(r'\[.*\]', line)[0]))
    return viz_ids


NET_TO_HSC = {'SPACEN_CNN224_FCONV': 'CNN224', 'SPACEN_CNN32_FCONV_S': 'CNN32', 'SPACEN_CNN28_FCONV': 'CNN28'}
DIM_TO_HSC = {49: 48, 64: 64, 784: 784}


class BaseRunner(object):
    def __init__(self):
        # warnings.simplefilter('error', UserWarning)
        self.args = self.argparse()
        if 'logdir_suffix' in vars(self.args):
            self.args.logdir += self.args.logdir_suffix
            del vars(self.args)['logdir_suffix']
        self.start = int(time.time())
        self.__backup, self.__bk_test = None, None
        self.__test = True

    def run(self):
        self.run_one(**vars(self.args))

    def run_one(
            self, batch_size, epochs, workers, final_dim, learning_rate, weight_decay,
            lr_sched_param, load, dataset, net, readme,
            datadir, normal_class, cuda, objective, logdir, viz_ids=(), **kwargs
    ):
        kwargs = dict(kwargs)
        acc_batches = kwargs.pop('acc_batches', 1)
        setup = trainer_setup(
            cuda, dataset, datadir, normal_class, net, final_dim, learning_rate, weight_decay, lr_sched_param, logdir,
            json.dumps({k: v for k, v in locals().items() if not k.endswith('results') and k not in ['setup', 'self']}),
            batch_size, workers, objective,
            self.start, **kwargs
        )
        try:
            trainer = ADTrainer(**setup)
            trainer.train(epochs, load, acc_batches=acc_batches)
            if self.__test and (epochs > 0 or load is not None):
                x = trainer.test(viz_ids)
            else:
                x = trainer.res
            setup['logger'].log_prints()
            return x
        except:
            setup['logger'].printlog += traceback.format_exc()
            setup['logger'].log_prints()  # no finally statement, because that breaks debugger
            raise

    @abstractmethod
    def add_parser_params(self, parser):
        pass

    def argparse(self, f_trans_args=None):
        from fcdd.util.argparse import ArgumentParser
        from fcdd.runners import Maintainer
        parser = ArgumentParser(description=__doc__)

        parser = self.add_parser_params(parser)

        if f_trans_args is not None:
            f_trans_args(parser)

        args = parser.parse_args()

        return Maintainer(
            args,
        )

    def get_base_logdir(self):
        return self.args.logdir.replace('{t}', time_format(self.start))

    def arg_to_ae(self, backup=True, restore=True):
        if restore:
            self.restore_args()
        if backup:
            self.backup_args()
        self.args.net = NET_TO_HSC[self.args.net]
        self.args.final_dim = DIM_TO_HSC[self.args.final_dim]
        self.args.objective = 'autoencoder'
        self.args.supervise_mode = 'unsupervised'
        if self.args.objective_params is None:
            self.args.objective_params = {}
        self.args.objective_params['heatmaps'] = None
        self.args.viz_ids = self.get_base_logdir()
        self.args.logdir += '_AE'

    def arg_to_hsc(self, backup=True, restore=True):
        if restore:
            self.restore_args()
        if backup:
            self.backup_args()
        self.args.net = NET_TO_HSC[self.args.net]
        self.args.objective = 'hard_boundary'
        if self.args.objective_params is None:
            self.args.objective_params = {}
        self.args.objective_params['heatmaps'] = 'grad'
        self.args.viz_ids = self.get_base_logdir()
        self.args.logdir += '_HSC'

    def restore_args(self):
        if self.__backup is not None:
            self.args = deepcopy(self.__backup)
            self.__test = self.__bk_test

    def backup_args(self):
        self.__backup = deepcopy(self.args)
        self.__bk_test = self.__test


class SeedsRunner(BaseRunner):
    def run(self):
        self.run_seeds(**vars(self.args))

    def run_seeds(
            self, batch_size, epochs, workers, final_dim, learning_rate, weight_decay,
            lr_sched_param, load, dataset, net, readme,
            datadir, normal_class, cuda, objective, it, logdir, **kwargs
    ):
        results = defaultdict(lambda: [])
        kwargs = dict(kwargs)
        its = range(it)
        if 'restrict_its' in kwargs:
            its = kwargs['restrict_its'] if kwargs['restrict_its'] is not None else its
            del kwargs['restrict_its']
        for i in its:
            this_logdir = pt.join(logdir, 'it_{}'.format(i))
            viz_ids = extract_viz_ids(self.args.viz_ids, normal_class, i)
            res = self.run_one(
                batch_size, epochs, workers, final_dim, learning_rate, weight_decay,
                lr_sched_param, load, dataset, net, readme,
                datadir, normal_class, cuda, objective, this_logdir, viz_ids=viz_ids,
                **{k: v for k, v in kwargs.items() if k not in ['viz_ids']}
            )
            for key in res:
                results[key].append(res[key])
        for key in results:
            plot_many_roc(
                logdir.replace('{t}', time_format(self.start)), results[key],
                labels=its, mean=True, name=key
            )
        return {key: mean_roc(results[key]) for key in results}

    @abstractmethod
    def add_parser_params(self, parser):
        pass

    def copy_files_to_super(self, *names):
        import os
        from shutil import copyfile
        sup = self.get_base_logdir()
        for (root, dirs, files) in os.walk(sup):
            for name in names:
                if name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join(sup, root.replace(sup, '')[1:].replace(os.sep, '___') + '___' + name)
                    copyfile(src, dst)
                    print('Copied {} to super.'.format(os.path.basename(dst)))


class ClassesRunner(SeedsRunner):
    def run(self):
        self.run_classes(**vars(self.args))

    def run_classes(
            self, batch_size, epochs, workers, final_dim, learning_rate, weight_decay,
            lr_sched_param, load, dataset, net, readme, datadir, cuda, objective, it,
            logdir, **kwargs
    ):
        results = defaultdict(lambda: [])
        kwargs = dict(kwargs)
        classes = range(no_classes(dataset))
        if 'cls_restrictions' in kwargs:
            classes = kwargs['cls_restrictions'] if kwargs['cls_restrictions'] is not None else classes
            del kwargs['cls_restrictions']
        for c in classes:
            cls_logdir = pt.join(logdir, 'normal_{}'.format(c))
            res = self.run_seeds(
                batch_size, epochs, workers, final_dim, learning_rate, weight_decay,
                lr_sched_param, load, dataset, net, readme,
                datadir, c, cuda, objective, it, cls_logdir, **kwargs
            )
            for key in res:
                results[key].append(res[key])
        for key in results:
            plot_many_roc(
                logdir.replace('{t}', time_format(self.start)), results[key],
                labels=str_labels(dataset), mean=True, name=key
            )
        return {key: mean_roc(results[key]) for key in results}

    @abstractmethod
    def add_parser_params(self, parser):
        pass
