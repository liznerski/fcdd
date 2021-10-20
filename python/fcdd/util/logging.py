import json
import os
import os.path as pt
import re
import sys
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from fcdd.util import DefaultList, CircleList, NumpyEncoder
from fcdd.util.metrics import mean_roc
from matplotlib import cm
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

MARKERS = ('.', 'x', '*', '+', 's', 'v', 'D', '1', 'p', '8', '2', '3', '4', '^', '<', '>', 'P', 'X', 'd', '|', '_')


def get_cmarker(totallen: int, lencolors=10):
    """ returns totallen many colored markers to use for different curves in one plot of matplotlib """
    if len(MARKERS) * lencolors < totallen:
        lencolors = totallen // len(MARKERS) + 1
    colors = cm.nipy_spectral(np.linspace(0, 1, lencolors)).tolist() * (totallen // lencolors + 1)
    markers = [[m] * lencolors for m in list(MARKERS)[:(totallen // lencolors + 1)]]
    markers = [dup for m in markers for dup in m]
    return list(zip(colors, markers))[:totallen]


def time_format(i: float) -> str:
    """ takes a timestamp (seconds since epoch) and transforms that into a datetime string representation """
    return datetime.fromtimestamp(i).strftime('%Y%m%d%H%M%S')


def colorize(imgs: List[Tensor], norm=True, rgb=True, cmap='jet') -> List[Tensor]:
    """
    For tensors of grayscaled images (n x 1 x h x w),
    colorizes each image of each tensor by using a colormap that maps [0, 1] -> [0, 1]^3.
    This is usually used to visualize heatmaps.
    :param imgs: tensor of grayscaled images (n x 1 x h x w)
    :param norm: whether to normalize each image s.t. min=0 and max=1
    :param rgb: whether the output tensor color channels are ordered by rgb (instead of bgr)
    :param cmap: the colormap that is used to colorize the images
    :return:
    """
    matplotlib.use('Agg')
    prev = None
    for img in imgs:
        if img is not None and len(img) > 0:
            assert prev is None or prev.dim() == img.dim()
            prev = img
    for j, img in enumerate(imgs):
        if img is not None and len(img) > 0:
            shp = list(img.shape)
            if shp[-3] == 1:
                shp[-3] = 3
            img = img.reshape(-1, *img.shape[-3:])
            if norm:
                img.sub_(
                    img.view(img.size(0), -1).min(1)[0][(..., ) + (None, ) * (img.dim() - 1)]
                ).div_(img.view(img.size(0), -1).max(1)[0][(..., ) + (None, ) * (img.dim() - 1)])  # scale each!
            img = torch.from_numpy(
                plt.get_cmap(cmap if rgb else '{}_r'.format(cmap))(img.cpu().detach().numpy())
            ).transpose(1, 4).squeeze(-1)[:, :-1].float()
            img = img.reshape(shp)
            imgs[j] = img
    return imgs


class Logger(object):
    """
    A customizable logger that is passed to Trainer instances to log data during training and testing.
    The .log method needs to be invoked at the end of every training iteration, i.e. each time after
    a batch of training data has been fully processed.
    All log data that has been gathered can be -- at any time -- written to a file using the .save method.
    The logger prints a part of the most current log data -- like loss, epoch, current batch -- with a
    predefined rate on the console.
    Furthermore, the logger offers various methods to write images, plots, and other types of log data
    that have not been gathered with the .log method, into the according log directory.
    For instance, this is used at the end of training to save some training heatmaps and after testing to
    save test heatmaps and ROC plots.
    """
    def __init__(self, logdir: str, fps=1, window=10, exp_start_time: float = None):
        """
        :param logdir: path to some directory, where all file storing methods write their files to,
            placeholder {t} is automatically replaced by the start time of the training
        :param fps: the rate in which the logger prints data on the console (frames per second)
        :param window: the memory depth of log data, i.e. the logger only
            maintains the top-k most recent batches in the .log method. At the end of an epoch the mean of
            those top-k batches is additionally saved.
        :param exp_start_time: start time of the overall training, just used for
            replacing {t} in logdir with it, defaults to the start time of this logger
        """
        self.start = int(time.time())
        self.exp_start_time = self.start if exp_start_time is None else exp_start_time
        self.dir = logdir.replace('{t}', time_format(self.exp_start_time))
        if not pt.exists(os.path.dirname(self.dir)):
            os.makedirs(os.path.dirname(self.dir))
        self.t = time.time()
        self.fps = fps
        self.history = defaultdict(DefaultList)
        self.history['err_all'] = CircleList(window)
        self.__window = window
        self.__lastepoch = 0
        self.__further_keys = []
        self.config_outfile = None
        self.logtxtfile = None
        self.loggingtxt = ''
        self.printlog = ''
        self.__warnings = []

    def reset(self, logdir: str = None, exp_start_time: float = None):
        """
        Resets all stored information. Also sets the start time
        :param logdir: sets a new logdir, defaults to None, which means keeping the old one
        :param exp_start_time: start time of the overall training, just used for
            replacing {t} in logdir with it (if logdir is not None), defaults to the old start time
        """
        self.start = int(time.time())
        self.exp_start_time = self.exp_start_time if exp_start_time is None else exp_start_time
        self.t = time.time()
        self.history = defaultdict(DefaultList)
        self.history['err_all'] = CircleList(self.__window)
        self.history['err'] = DefaultList()
        self.__lastepoch = 0
        self.__further_keys = []
        self.logtxtfile = None
        self.loggingtxt = ''
        self.log_prints()
        self.__warnings = []
        if logdir is not None:
            self.dir = logdir.replace('{t}', time_format(self.exp_start_time))
            if not pt.exists(os.path.dirname(self.dir)):
                os.makedirs(os.path.dirname(self.dir))

    def log(self, epoch: int, nbat: int, batches: int, err: Tensor, info: dict = None, infoprint="", force_print=False):
        """
        Logs data of a training iteration. Maintains achieved losses and other metrics (info) in a CircleList.
        Per epoch it also stores the last average loss (and other metrics) that has been logged.
        Prints current epoch, current batch index, running average loss, and a given info string on the console.
        Print is skipped if it exceeds the number of print messages per second
        that are allowed by the fps parameter.
        :param epoch: current epoch
        :param nbat: current batch index
        :param batches: the number of batches per epoch
        :param err: the computed loss of this iteration, Tensor containing a single scalar
        :param info: dictionary of further metrics of this iteration {str -> Tensor}.
            For each key in this dictionary a CircleList is created that is maintained and handled just like the loss.
        :param infoprint: a string that is to be printed in addition to the usual log data.
        :param force_print: force a print, e.g. at the end of an epoch, ignoring the fps constraint
        """
        if info is not None:
            self.log_info(info)

        def save_epoch(ep):
            self.history['err'][ep] = np.mean(self.history['err_all'])
            self.history['err_std'][ep] = np.std(self.history['err_all'])
            for k in self.__further_keys:
                self.history[k][ep] = np.mean(self.history['{}_all'.format(k)])

        diff = time.time() - self.t
        if diff > 1/self.fps or force_print:
            self.t = time.time()
            save_epoch(epoch)
            self.print(
                'EPOCH {:02d} NBAT {:04d}/{:04d} ERR {:01f} {} INFO {}'
                .format(
                    epoch, nbat, batches,
                    self.history['err'][epoch],
                    ' '.join([
                        '{:4} {:01f}'.format(k.upper()[:13], self.history[k][epoch]) for k in self.__further_keys
                    ]),
                    infoprint
                )
            )
        if epoch > self.__lastepoch:  # force final update of last epoch at first batch of new epoch
            save_epoch(epoch-1)

        self.history['err_all'].append(err.data.item())
        self.__lastepoch = epoch

    def log_info(self, info: dict, epoch: int = None):
        """
        Logs a dictionary of metrics (unique name -> scalar value) {str -> Tensor} in CircleLists.
        Does not compute an average at the end of an epoch. This is done in the .log method.
        :param info: dictionary of metrics that are to be maintained like the loss.
        :param epoch: current epoch
        """
        for k, v in info.items():
            if k not in self.__further_keys:
                if '{}_all'.format(k) in self.history:
                    raise ValueError('{} is already part of the history.'.format(k))
                self.history['{}_all'.format(k)] = CircleList(self.__window)
                self.__further_keys.append(k)
            self.history['{}_all'.format(k)].append(v.data.item())
            if epoch is not None:
                self.history[k][epoch] = np.mean(self.history['{}_all'.format(k)])

    def print(self, txt: str, fps: bool = False, err: bool = False):
        """
        Prints text on the console.
        All prints are remembered and can be written to a file by invoking the .log_prints method.
        :param txt: the text that is to be printed
        :param fps: whether to ignore the fps constraint
        :param err: whether to print on the stderr stream instead
        """
        if not fps:
            print(txt, file=sys.stderr if err else sys.stdout)
            self.printlog += '{}\n'.format(txt)
        else:
            diff = time.time() - self.t
            if diff > 1 / self.fps:
                self.t = time.time()
                print(txt, file=sys.stderr if err else sys.stdout)
                self.printlog += '{}\n'.format(txt)

    def log_prints(self):
        """
        Writes all remembered prints to a file named print.log in the log directory.
        Afterwards, empties the collection of remembered prints.
        """
        outfile = pt.join(self.dir, 'print.log')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        with open(outfile, 'a') as writer:
            writer.write(self.printlog)
        self.printlog = ''

    def save(self, subdir='.'):
        """
        Writes all data logged with the .log and .log_info method to a file in the log directory.
        That is the history of losses and metrics.
        Also writes the start time, the duration of the so far training, and the text that has been
        logged via the .logtxt method.
        The .save method should be invoked at the end of the training/testing.
        :param subdir: if given, creates a subdirectory in the log directory. The data is written to a file
            in this subdirectory instead.
        """
        outfile = pt.join(self.dir, subdir, 'history.json')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        with open(outfile, 'w') as writer:
            json.dump(self.history, writer)
        outfile = pt.join(self.dir, subdir, 'log.txt')
        self.logtxtfile = outfile
        txt = self.loggingtxt
        self.loggingtxt = ''
        txt += 'START: {} \n'.format(
            datetime.fromtimestamp(self.start).strftime('%d-%m-%Y %H:%M:%S')
        )
        txt += 'DURATION: {} \n'.format(
            datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(self.start)
        )
        with open(outfile, 'w') as writer:
            writer.write(txt)

    def single_save(self, name: str, dic: Any, subdir='.'):
        """
        Writes a given dictionary to a json file in the log directory.
        Returns without impact if the size of the dictionary exceeds 10MB.
        :param name: name of the json file
        :param dic: serializable dictionary
        :param subdir: if given, creates a subdirectory in the log directory. The data is written to a file
            in this subdirectory instead.
        """
        outfile = pt.join(self.dir, subdir, '{}.json'.format(name))
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        if isinstance(dic, dict):
            sz = np.sum([sys.getsizeof(v) for k, v in dic.items()])
            if sz > 10000000:
                self.logtxt(
                    'WARNING: Could not save {}, because size of dict is {}, which exceeded 10MB!'
                    .format(pt.join(self.dir, subdir, '{}.json'.format(name)), sz),
                    print=True
                )
                return
            with open(outfile, 'w') as writer:
                json.dump(dic, writer, cls=NumpyEncoder)
        else:
            torch.save(dic, outfile.replace('.json', '.pth'))

    def plot(self, subdir='.'):
        """
        Plots logged loss and metrics (info) and writes the plots to pdf files in the log directory.
        The .plot method should be invoked at the end of the training/testing.
        :param subdir: if given, creates a subdirectory in the log directory. The data is written to a file
            in this subdirectory instead.
        :return:
        """
        matplotlib.use('Agg')
        outfile = pt.join(self.dir, subdir, 'err.pdf')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        plt.plot(self.history['err'], ls='-')
        legend = ["err"]
        if 'val_err' in self.__further_keys:
            plt.plot(self.history['val_err'], ls='-')
            legend += ['val_err']
        plt.legend(legend)
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.savefig(outfile, format='pdf')
        plt.close()
        for k in self.__further_keys:
            if k == 'val_err':
                continue
            plt.plot(self.history[k], ls='-')
            plt.ylabel(k)
            plt.xlabel('epoch')
            plt.savefig(outfile.replace('err.pdf', '{}.pdf'.format(k)))
            plt.close()

    def single_plot(self, name: str, values: List[float], xs: List[float] = None,
                    xlabel: str = None, ylabel: str = None, legend: List = (), subdir='.'):
        """
        Plots given values and writes the plot to a pdf file in the log directory.
        :param name: the name of the pdf file
        :param values: the values to plot on the y-axis
        :param xs: the corresponding values on the x-axis, defaults to [1,...,n]
        :param xlabel: the x-axis label
        :param ylabel: the y-axis label
        :param legend: a legend that is to printed in a corner of the plot
        :param subdir: if given, creates a subdirectory in the log directory. The data is written to a file
            in this subdirectory instead.
        :return:
        """
        matplotlib.use('Agg')
        outfile = pt.join(self.dir, subdir, '{}.pdf'.format(name))
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        if xs is None:
            plt.plot(values)
        else:
            plt.plot(xs, values)
        plt.legend(legend)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.savefig(outfile, format='pdf')
        plt.close()

    def imsave(self, name: str, tensors: Tensor, subdir='.', nrow=8, scale_mode='each',
               rowheaders: List[str] = None, pad=2, row_sep_at: Tuple[int, int] = (None, None),
               colcounter: List[str] = None):
        """
        Interprets a tensor (n x c x h x w) as a grid of images and writes this to a png file.
        :param name: the name of the png file
        :param tensors: the tensor of images
        :param subdir: if given, creates a subdirectory in the log directory. The data is written to a file
            in this subdirectory instead.
        :param nrow: the number of images per row in the png
        :param scale_mode: the type of normalization. Either "none" for no normalization or "each" to
            scale each image individually, s.t. it lies exactly in the range [0, 1].
        :param rowheaders: a list of headers for the rows.
            Each element of the list is printed in front of its corresponding row in the png.
            The method expects less than 6 characters per header. More characters might be printed over
            the actual images. Defaults to None, where no headers are printed.
        :param pad: the amount of padding that is added in between images in the grid.
        :param row_sep_at: two integer values or empty tuple. If it contains two integers, it adds
            an additional row of zeros that acts as a separator between rows. The first value describes
            the height of the separating row and the second value the position (e.g. 2 to put in between the
            first and second row).
        :param colcounter: a list of headers for the columns.
            Each element of the list is printed in front of its corresponding column in the png.
            Defaults to None for no column headers.
        :return:
        """
        outfile = pt.join(self.dir, subdir, '{}.png'.format(name))
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        t = tensors.clone()
        if scale_mode != 'none':
            zero_std_mask = (tensors.reshape(tensors.size(0), -1).std(1) == 0)

        if scale_mode == 'each':
            t = vutils.make_grid(t, nrow=nrow, scale_each=True, normalize=True, padding=pad)
        elif scale_mode == 'none':
            t = vutils.make_grid(t, nrow=nrow, scale_each=False, normalize=False, padding=pad)
        else:
            raise NotImplementedError('scale mode {} not known'.format(scale_mode))

        if scale_mode != 'none':
            zero_std_mask_grid = vutils.make_grid(
                zero_std_mask[:, None, None, None].repeat((1, ) + tuple(tensors.shape[1:])),
                nrow=nrow, scale_each=False, normalize=False, padding=pad
            )
            vals = vutils.make_grid(
                tensors, nrow=nrow, scale_each=False, normalize=False, padding=pad
            )
            t[zero_std_mask_grid] = vals[zero_std_mask_grid]
            del zero_std_mask_grid, zero_std_mask, vals

        t = t.transpose(0, 2).transpose(0, 1).numpy() * 255
        if rowheaders is not None:
            n, c, h, w = tensors.shape
            t = np.concatenate((torch.zeros(t.shape[0], int(w * 1.8), 3), t), 1)  # add black front column
            for i, head in enumerate(rowheaders):
                if len(str(head)) > 6:
                    import warnings
                    warnings.warn(
                        'Header for image {} is too large, some content will be printed on actual image!'.format(name)
                    )
                sc = 0.5 + 0.5 * (tensors.shape[-1] // 40)
                th = 1 + 1 * (tensors.shape[-1] // 100)
                t = cv2.putText(
                    t, str(head), (0, h - 10 * th + (h + 2) * i),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th
                )
        if colcounter is not None:
            n, c, h, w = tensors.shape
            t = np.concatenate((torch.zeros(32, t.shape[1], 3), t), 0)  # add black front row
            for i, s in enumerate(colcounter):
                t = cv2.putText(
                    t, str(s), (w - 24 + (w + 2) * i, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
                )

        if row_sep_at is not None and row_sep_at[0] is not None and len(row_sep_at) == 2:
            height, at = row_sep_at
            t = np.concatenate([t[:at], np.zeros([height, t.shape[1], t.shape[2]]), t[at:]]).astype(np.float32)

        if t.shape[-1] == 3:
            t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outfile, t)

    def snapshot(self, net: torch.nn.Module, opt: Optimizer, sched: _LRScheduler = None, epoch: int = None, subdir='.'):
        """
        Writes a snapshot of the training, i.e. network weights, optimizer state and scheduler state to a file
        in the log directory.
        :param net: the neural network
        :param opt: the optimizer used
        :param sched: the learning rate scheduler used
        :param epoch: the current epoch
        :param subdir: if given, creates a subdirectory in the log directory. The data is written to a file
            in this subdirectory instead.
        :return:
        """
        outfile = pt.join(self.dir, subdir, 'snapshot.pt')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        torch.save(
            {'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch}
            , outfile
        )
        return outfile

    def save_params(self, net: torch.nn.Module, params: str, subdir='.'):
        """
        Writes a string representation of the network and all given parameters as text to a
        configuration file named config.txt in the log directory.
        Also saves a compression of the complete current code as src.tar.gz in the log directory.
        :param net: the neural network
        :param params: all parameters of the training in form of a string representation (json dump of a dictionary)
        :param subdir: suffix to append to logdir
        """
        outfile = pt.join(self.dir, subdir, 'config.txt')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        self.config_outfile = outfile
        with open(outfile, 'w') as writer:
            writer.write("{}\n\n{}".format(net, params))

        def filter(tarinfo):
            exclude = re.compile('(.*__pycache__.*)|(.*{}.*)'.format(os.sep+'venv'+os.sep))
            if not exclude.fullmatch(tarinfo.name):
                return tarinfo
            else:
                return None

        outfile = pt.join(self.dir, subdir, 'src.tar.gz')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        with tarfile.open(outfile, "w:gz") as tar:
            root = pt.join(pt.dirname(__file__), '..')
            tar.add(root, arcname=os.path.basename(root), filter=filter)

        self.print('Successfully saved code at {}'.format(outfile), fps=False)

    def logtxt(self, s: str, print=False):
        """
        Either appends txt directly to existing file named log.txt (created in the .save method),
        or, if not yet existent, memorizes input and writes to log.txt when save() is executed.
        :param s: string that is to be logged in the log.txt file in the log directory
        :param print: whether to also print the string on the console
        """
        if print:
            self.print(s)
        if self.logtxtfile is None:
            self.loggingtxt += '{}\n'.format(s)
        else:
            with open(self.logtxtfile, 'a') as writer:
                writer.write('{}\n'.format(s))

    def warning(self, s: str, unique: bool = False, print: bool = True):
        """
        Writes a warning to the WARNING.log file in the log directory.
        :param s: the warning that is to be written
        :param unique: whether a warning that has already been written is to be ignored
        :param print: whether to additionally print the warning on the console
        """
        if unique and s in self.__warnings:
            return
        if print:
            self.print(s, err=True)
        outfile = pt.join(self.dir, 'WARNINGS.log')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        with open(outfile, 'a') as writer:
            writer.write(s)
        self.__warnings.append(s)

    def timeit(self, msg: str = 'Operation'):
        """
        Returns a Timer that is to be used in a `with` statement to measure the time that all operations inside
        the `with` statement took. Once the `with` statement is exited, prints the measured time together with msg.
        """
        return self.Timer(self, msg)

    class Timer(object):
        def __init__(self, logger, msg):
            self.logger = logger
            self.msg = msg
            self.start = None

        def __enter__(self):
            self.start = time.time()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.print('{} took {} seconds.'.format(self.msg, time.time() - self.start))


def plot_many_roc(logdir: str, results: List[dict], labels: List[str] = None, name: str = 'roc', mean: bool = False):
    """
    Plots the ROCs of different training runs together in one plot and writes that to a pdf file in the log directory.
    The ROCs are given in form of result dictionaries {'tpr': [], 'fpr': [], 'ths': [], 'auc': int, ...},
    where ths contains the thresholds, tpr the true positive rates per threshold, fpr the false positive rates
    per threshold and auc the AuROC of the curve.
    :param logdir: the log directory in which the pdf file is to be stored
    :param results: a list of result dictionaries
    :param labels: a list of labels for the individual ROCs
    :param name: the name of the pdf file
    :param mean: whether to also plot a dotted "mean ROC"
    """
    if results is None or any([r is None for r in results]) or len(results) == 0:
        return None
    matplotlib.use('Agg')
    outfile = pt.join(logdir, '{}.pdf'.format(name))
    if not pt.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    if labels is None:
        labels = list(range(len(results)))
    legend = []
    x = 'fpr' if 'fpr' in results[0] else 'recall'
    y = 'tpr' if 'tpr' in results[0] else 'prec'
    for c, res in enumerate(results):
        plt.plot(res[x], res[y], linewidth=0.5)
        legend.append('{} {:5.2f}%'.format(labels[c], res['auc']*100))
    if mean:
        mean_res = mean_roc(results)
        plt.plot(mean_res[x], mean_res[y], '--', linewidth=1)
        legend.append('{} {:5.2f}%'.format('mean', mean_res['auc'] * 100))
    plt.legend(legend,  fontsize='xx-small' if len(legend) > 20 else 'x-small')
    plt.savefig(outfile, format='pdf')
    plt.close()

