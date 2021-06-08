import json
import os.path as pt
import re
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from argparse import ArgumentParser

from fcdd.datasets import no_classes, str_labels
from fcdd.runners.argparse_configs import DefaultConfig
from fcdd.training.setup import trainer_setup
from fcdd.training.super_trainer import SuperTrainer
from fcdd.util.logging import plot_many_roc, time_format
from fcdd.util.metrics import mean_roc


def extract_viz_ids(dir: str, cls: str, it: int):
    """
    Extracts the indices of images that have been logged in an old experiment.
    :param dir: directory that contains log data of the old experiment
    :param cls: class that is considered to be nominal
    :param it: the iteration (random seed)
    :return: [(indices for label 0..), (indices for label 1..)]
    """
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


NET_TO_HSC = {
    'FCDD_CNN224': 'CNN224',  'FCDD_CNN224_W': 'CNN224', 'FCD_CNN224_VGG_F': 'CNN224_VGG_F',
    'FCDD_CNN224_VGG_NOPT': 'CNN224_VGG_NOPT_1000', 'FCDD_CNN224_VGG': 'CNN224_VGG',
    'FCDD_CNN32_LW3K': 'CNN32_LW3K', 'FCDD_CNN32_S': 'CNN32',
    'FCDD_CNN28_W': 'CNN28_W', 'FCDD_CNN28': 'CNN28',
}
NET_TO_AE = {
    'FCDD_CNN224': 'AE224', 'FCDD_CNN224_W': 'AE224', 'FCD_CNN224_VGG_F': 'AE224_VGG_F',
    'FCDD_CNN224_VGG_NOPT': 'AE224_VGG_NOPT', 'FCDD_CNN224_VGG': 'AE224_VGG',
    'FCDD_CNN32_LW3K': 'AE32_LW3K', 'FCDD_CNN32_S': 'AE32',
    'FCDD_CNN28_W': 'AE28', 'FCDD_CNN28': 'AE28',
}


class BaseRunner(object):
    def __init__(self, config: DefaultConfig):
        """
        Basic runner that runs a training for one seed and one specific class considered nominal.
        To run an experiment, create an instance of the runner with an argument configuration object
        and invoke the run method.
        :param config: an argument configuration object that adds all arguments, defined in runners/argparse_configs
        """
        self.args = ArgumentParser(
            description=
            "Train a neural network module as explained in the `Explainable Deep Anomaly Detection` paper. "
            "Based on the objective, train either FCDD or one of the baselines presented in the paper, i.e. "
            "an autoencoder or the HSC objective. Log achieved scores, metrics, plots, and heatmaps "
            "for both test and training data. "
        )
        self.args = config(self.args)
        self.args = self.args.parse_args()
        if 'logdir_suffix' in vars(self.args):
            self.args.logdir += self.args.logdir_suffix
            del vars(self.args)['logdir_suffix']
        self.start = int(time.time())
        self.__backup, self.__bk_test = None, None
        self._test = True

    def run(self):
        self.run_one(**vars(self.args))

    def run_one(
            self, viz_ids: tuple, **kwargs
            # kwargs should contain all parameters of the setup function in training.setup
    ):
        kwargs = dict(kwargs)
        readme = kwargs.pop('readme', '')
        kwargs['log_start_time'] = self.start
        kwargs['viz_ids'] = viz_ids
        kwargs['config'] = '{}\n\n{}'.format(json.dumps(kwargs), readme)
        acc_batches = kwargs.pop('acc_batches', 1)
        epochs = kwargs.pop('epochs')
        load = kwargs.pop('load', None)
        kwargs.pop('viz_ids')
        setup = trainer_setup(
            **kwargs
        )
        try:
            trainer = SuperTrainer(**setup)
            trainer.train(epochs, load, acc_batches=acc_batches)
            if self._test and (epochs > 0 or load is not None):
                x = trainer.test(viz_ids)
            else:
                x = trainer.res
            setup['logger'].log_prints()
            return x
        except:
            setup['logger'].printlog += traceback.format_exc()
            setup['logger'].log_prints()  # no finally statement, because that breaks debugger
            raise

    def get_base_logdir(self):
        """ returns the actualy log directory """
        return self.args.logdir.replace('{t}', time_format(self.start))

    def arg_to_ae(self, backup=True, restore=True):
        """ transfers a part of the parameters to train an autoencoder instead, with reconstruction loss heatmaps """
        if restore:
            self.restore_args()
        if backup:
            self.backup_args()
        self.args.net = NET_TO_AE[self.args.net]
        self.args.objective = 'autoencoder'
        self.args.supervise_mode = 'unsupervised'
        self.args.blur_heatmaps = True
        self.args.viz_ids = self.get_base_logdir()
        self.args.logdir += '_AE'

    def arg_to_hsc(self, backup=True, restore=True):
        """ transfers a part of the parameters to train the HSC objective instead, with gradient heatmaps """
        if restore:
            self.restore_args()
        if backup:
            self.backup_args()
        self.args.net = NET_TO_HSC[self.args.net]
        self.args.objective = 'hsc'
        self.args.blur_heatmaps = True
        self.args.viz_ids = self.get_base_logdir()
        self.args.logdir += '_HSC'

    def restore_args(self):
        if self.__backup is not None:
            self.args = deepcopy(self.__backup)
            self._test = self.__bk_test

    def backup_args(self):
        self.__backup = deepcopy(self.args)
        self.__bk_test = self._test


class SeedsRunner(BaseRunner):
    """
    Runner that runs a training for multiple random seeds and one specific class considered nominal.
    Also, creates a combined AuROC graph that contains curves for all seeds at once.
    The log directory will contain the combined graph and a subdirectory for each seed.
    The subdirectories are named `it_x`, with x being the number of the seed
    (e.g. 2 for the second seed/training_iteration).
    """
    def run(self):
        self.run_seeds(**vars(self.args))

    def run_seeds(
            self, it: int, **kwargs
    ):
        results = defaultdict(lambda: [])
        kwargs = dict(kwargs)
        logdir = kwargs.pop('logdir')
        viz_ids = kwargs.pop('viz_ids')
        its = range(it)
        if 'its_restrictions' in kwargs:
            its = kwargs['its_restrictions'] if kwargs['its_restrictions'] is not None else its
            del kwargs['its_restrictions']
        for i in its:
            kwargs['logdir'] = pt.join(logdir, 'it_{}'.format(i))
            this_viz_ids = extract_viz_ids(viz_ids, kwargs['normal_class'], i)
            res = self.run_one(
                this_viz_ids, **kwargs
            )
            for key in res:
                results[key].append(res[key])
        for key in results:
            plot_many_roc(
                logdir.replace('{t}', time_format(self.start)), results[key],
                labels=its, mean=True, name=key
            )
        return {key: mean_roc(results[key]) for key in results}


class ClassesRunner(SeedsRunner):
    """
    Runner that runs multiple trainings with different seeds for each class of the dataset considered nominal.
    Also, creates a combined AUC graph that contains curves for all classes at once.
    The log directory will contain the combined graph and a subdirectory for each class.
    The subdirectories are named `normal_x`, with x being the class id.
    Each class directory will further contain combined roc curves for all seeds and a subdirectory for each seed.
    """
    def run(self):
        self.run_classes(**vars(self.args))

    def run_classes(
            self, **kwargs
    ):
        results = defaultdict(lambda: [])
        kwargs = dict(kwargs)
        it = kwargs.pop('it')
        logdir = kwargs['logdir']
        classes = range(no_classes(kwargs['dataset']))
        if 'cls_restrictions' in kwargs:
            classes = kwargs['cls_restrictions'] if kwargs['cls_restrictions'] is not None else classes
            del kwargs['cls_restrictions']
        for c in classes:
            cls_logdir = pt.join(logdir, 'normal_{}'.format(c))
            kwargs['logdir'] = cls_logdir
            kwargs['normal_class'] = c
            try:
                res = self.run_seeds(
                   it, **kwargs
                )
                for key in res:
                    results[key].append(res[key])
            finally:
                print('Plotting ROC for completed classes up to {}...'.format(c))
                for key in results:
                    plot_many_roc(
                        logdir.replace('{t}', time_format(self.start)), results[key],
                        labels=str_labels(kwargs['dataset']), mean=True, name=key
                    )
        for key in results:
            plot_many_roc(
                logdir.replace('{t}', time_format(self.start)), results[key],
                labels=str_labels(kwargs['dataset']), mean=True, name=key
            )
        return {key: mean_roc(results[key]) for key in results}

