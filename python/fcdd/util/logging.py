import json
import os
import os.path as pt
import re
import sys
import tarfile
import time
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from fcdd.util import DefaultList, CircleList, NumpyEncoder
from fcdd.util.metrics import mean_roc
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

MARKERS = ('.', 'x', '*', '+', 's', 'v', 'D', '1', 'p', '8', '2', '3', '4', '^', '<', '>', 'P', 'X', 'd', '|', '_')


def get_cmarker(totallen, lencolors=10):
    if len(MARKERS) * lencolors < totallen:
        lencolors = totallen // len(MARKERS) + 1
    colors = cm.nipy_spectral(np.linspace(0, 1, lencolors)).tolist() * (totallen // lencolors + 1)
    markers = [[m] * lencolors for m in list(MARKERS)[:(totallen // lencolors + 1)]]
    markers = [dup for m in markers for dup in m]
    return list(zip(colors, markers))[:totallen]


def time_format(i):
    return datetime.fromtimestamp(i).strftime('%Y%m%d%H%M%S')


def scale_row(tensors, nrow, inplace=True):
    rows = int(np.ceil(tensors.size(0) / nrow))
    if not inplace:
        tensors = tensors.clone()
    for r in range(rows):
        s = r * nrow
        tensors[s:s+nrow].sub_(tensors[s:s+nrow].min()).div_(tensors[s:s+nrow].max())
    return tensors


def scale_type(types, inplace=True):
    if not inplace:
        types = [typ.clone() if typ is not None and len(typ) > 0 else typ for typ in types]
    for typ in types:
        if typ is not None and len(typ) > 0:
            typ.sub_(typ.min()).div_(typ.max())
    return types


def log_scale_each(tensors, inplace=True, max_before_log=10, eps=1e-14):
    if not inplace:
        tensors = tensors.clone()
    mins = tensors.view(tensors.size(0), -1).min(1)[0][:, None, None, None]
    tensors.sub_(mins)
    maxs = tensors.view(tensors.size(0), -1).max(1)[0][:, None, None, None]
    tensors.div_(maxs + eps)
    tensors = (tensors * max_before_log + 1).log()
    maxs = tensors.view(tensors.size(0), -1).max(1)[0][:, None, None, None]
    tensors.div_(maxs + eps)
    return tensors


def colorize(imgs, norm=True, rgb=True, cmap='jet'):
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
                plt.get_cmap(cmap if rgb else '{}_r'.format(cmap))(img)
            ).transpose(1, 4).squeeze(-1)[:, :-1].float()
            img = img.reshape(shp)
            imgs[j] = img
    return imgs


def equal_channel(ten, chpos=1):
    ten = ten.transpose(0, chpos)
    prev = ten[0]
    for c in range(ten.size(0)):
        if (prev != ten[c]).sum() > 0:
            return False
    return True


class Logger(object):
    """
    Logger that can be used for logging training stuff.
    The .log method needs to be invoked at the end of every training iteration.
    Training information, like loss and metrics, are stored and can be written to a file
    at any time using the .save method.
    Also prints current most important info with a predefined rate on the console.
    :param logdir - str: path to some directory, where all file storing methods store their files
        placeholder {t} is automatically replaced by starttime
    :param fps - int: cap the print messages to frames per second
    :param window - int: the memory depth of the metrics information, i.e.
        maintains the top-k most recent batches.
        At the end of each epoch this window is read for all metrics and a mean is stored.
    :param exp_start_time - int: starting time of the overall experiment, just used for
        replacing {t} in logdir with it, defaults to the start time of this logger
    """
    def __init__(self, logdir, fps=1, window=10, exp_start_time=None):
        self.start = int(time.time())
        self.exp_start_time = self.start if exp_start_time is None else exp_start_time
        self.t = time.time()
        self.dir = logdir.replace('{t}', time_format(self.exp_start_time))
        if not pt.exists(os.path.dirname(self.dir)):
            os.makedirs(os.path.dirname(self.dir))
        self.fps = fps
        self.eps = 1
        self.epos = 0
        self.history = defaultdict(DefaultList)
        self.history['err_all'] = CircleList(window)
        self.__window = window
        self.__ittime = CircleList(window * 5)
        self.__lastbat = 0
        self.__lastepoch = 0
        self.__further_keys = []
        self.__full_time_estimate = None
        self.config_outfile = None
        self.logtxtfile = None
        self.loggingtxt = ''
        self.printlog = ''
        self.__warnings = []

    def reset(self, logdir=None):
        """
        Resets all stored information.
        :param logdir: sets a new logdir, defaults to None, which means keeping the old one
        :return:
        """
        self.start = int(time.time())
        self.t = time.time()
        self.history = defaultdict(DefaultList)
        self.history['err_all'] = CircleList(self.__window)
        self.history['err'] = DefaultList()
        self.__ittime = CircleList(self.__window * 5)
        self.__lastbat = 0
        self.__lastepoch = 0
        self.__further_keys = []
        self.logtxtfile = None
        self.loggingtxt = ''
        self.log_prints()
        self.__warnings = []
        if logdir is not None:
            self.dir = logdir

    def log(self, epoch, nbat, batches, err, info=None, infoprint="", force_print=False):
        if info is not None:
            self.log_info(info)

        def save_epoch(ep):
            self.history['err'][ep] = np.mean(self.history['err_all'])
            self.history['err_std'][ep] = np.std(self.history['err_all'])
            for k in self.__further_keys:
                self.history[k][ep] = np.mean(self.history['{}_all'.format(k)])

        diff = time.time() - self.t
        if diff > 1/self.fps or force_print:
            batdiff = nbat - self.__lastbat
            batdiff = batdiff if batdiff > 0 else batches + nbat - self.__lastbat
            self.__lastbat = nbat
            self.__ittime.append(diff / batdiff)
            self.t = time.time()
            save_epoch(epoch)
            self.print(
                'EPOCH {:02d} NBAT {:04d}/{:04d} ERR {:01f} INFO {}'
                .format(
                    epoch, nbat, batches,
                    self.history['err'][epoch],
                    infoprint
                )
            )
        if epoch > self.__lastepoch:  # force final update of last epoch at first batch of new epoch
            save_epoch(epoch-1)

        self.history['err_all'].append(err.data.item())
        self.__lastepoch = epoch

    def log_info(self, info, epoch=None):
        for k, v in info.items():
            if k not in self.__further_keys:
                if '{}_all'.format(k) in self.history:
                    raise ValueError('{} is already part of the history.'.format(k))
                self.history['{}_all'.format(k)] = CircleList(self.__window)
                self.__further_keys.append(k)
            self.history['{}_all'.format(k)].append(v.data.item())
            if epoch is not None:
                self.history[k][epoch] = np.mean(self.history['{}_all'.format(k)])

    def print(self, txt, fps=False, err=False):
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
        outfile = pt.join(self.dir, 'print.log')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        with open(outfile, 'a') as writer:
            writer.write(self.printlog)
        self.printlog = ''

    def save(self, suffix='.'):
        outfile = pt.join(self.dir, suffix, 'history.json')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        with open(outfile, 'w') as writer:
            json.dump(self.history, writer)
        outfile = pt.join(self.dir, suffix, 'log.txt')
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

    def single_save(self, name, dic, suffix='.'):
        outfile = pt.join(self.dir, suffix, '{}.json'.format(name))
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        if isinstance(dic, dict):
            sz = np.sum([sys.getsizeof(v) for k, v in dic.items()])
            if sz > 10000000:
                self.logtxt(
                    'WARNING: Could not save {}, because size of dict is {}, which exceeded 10MB!'
                    .format(pt.join(self.dir, suffix, '{}.json'.format(name)), sz),
                    print=True
                )
                return
            with open(outfile, 'w') as writer:
                json.dump(dic, writer, cls=NumpyEncoder)
        elif isinstance(dic, torch.Tensor):
            torch.save(dic, outfile.replace('.json', '.pth'))

    def plot(self, suffix='.'):
        matplotlib.use('Agg')
        outfile = pt.join(self.dir, suffix, 'err.pdf')
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

    def single_plot(self, name, values, xs=None, xlabel=None, ylabel=None, legend=(), suffix='.'):
        matplotlib.use('Agg')
        outfile = pt.join(self.dir, suffix, '{}.pdf'.format(name))
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

    def imsave(self, name, tensors, suffix='.', nrow=8, scale_mode='each',
               rowheaders=None, pad=2, row_sep_at=(), colcounter=None):
        outfile = pt.join(self.dir, suffix, '{}.png'.format(name))
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        t = tensors.clone()
        if scale_mode != 'none':
            zero_std_mask = (tensors.reshape(tensors.size(0), -1).std(1) == 0)

        if scale_mode == 'each':
            t = vutils.make_grid(t, nrow=nrow, scale_each=True, normalize=True, padding=pad)
        elif scale_mode == 'row':
            t = scale_row(t, nrow)
            t = vutils.make_grid(t, nrow=nrow, scale_each=False, normalize=False, padding=pad)
        elif scale_mode == 'none':
            t = vutils.make_grid(t, nrow=nrow, scale_each=False, normalize=False, padding=pad)
        elif scale_mode == 'log_each':
            t = log_scale_each(t)
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

        if row_sep_at is not None and len(row_sep_at) == 2:
            height, at = row_sep_at
            t = np.concatenate([t[:at], np.zeros([height, t.shape[1], t.shape[2]]), t[at:]]).astype(np.float32)

        if t.shape[-1] == 3:
            t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outfile, t)

    def plt_imsave(self, name, mat, suffix='.'):
        matplotlib.use('Agg')
        outfile = pt.join(self.dir, suffix, '{}.pdf'.format(name))
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        plt.imshow(mat, cmap='RdBu')
        plt.axis('off')
        plt.colorbar()
        for (j, i), label in np.ndenumerate(mat):
            label = '{:.2f}'.format(label)
            plt.text(i, j, label, ha='center', va='center')
            plt.text(i, j, label, ha='center', va='center')
        plt.savefig(outfile)
        plt.close()

    def snapshot(self, net, opt, sched=None, epoch=None, c=None, suffix='.'):
        outfile = pt.join(self.dir, suffix, 'snapshot.pt')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        torch.save(
            {'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch, 'c': c}
            , outfile
        )
        return outfile

    def save_params(self, net, params, pt_net=None, suffix='.', further=None):
        """
        Saves all given params as text in configfile.
        Also saves complete code.
        :param net: network model
        :param params: argparse parameter dict
        :param pt_net: pretrain network model
        :param suffix: suffix to append to logdir
        :param further: further information in form of a dict
        :return:
        """
        outfile = pt.join(self.dir, suffix, 'config.txt')
        if not pt.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        if further is None:
            further = ""
        self.config_outfile = outfile
        with open(outfile, 'w') as writer:
            writer.write("{}\n\n{}\n\n{}\n\n{}".format(net, pt_net, params, further))

        def filter(tarinfo):
            exclude = re.compile('(.*__pycache__.*)|(.*{}.*)'.format(os.sep+'venv'+os.sep))
            if not exclude.fullmatch(tarinfo.name):
                return tarinfo
            else:
                return None

    def logtxt(self, s, print=False):
        """
        Either writes txt directly to existing logtxtfile (created in .save()).
        Or if not yet existent, memorized input and stores when save() is executed.
        :param s:
        :return:
        """
        if print:
            self.print(s)
        if self.logtxtfile is None:
            self.loggingtxt += '{}\n'.format(s)
        else:
            with open(self.logtxtfile, 'a') as writer:
                writer.write('{}\n'.format(s))

    def timeit(self, msg='Operation'):
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

    def warning(self, s, unique=False, print=True):
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


class ItLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history['it_err'] = DefaultList()
        self.__further_it_keys = []

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.history['it_err'] = DefaultList()

    def log(self, epoch, nbat, batches, err, info=None, infoprint="", force_print=False):
        super().log(epoch, nbat, batches, err, info, infoprint, force_print)
        self.history['it_err'].append(err.data.item())

    def log_info(self, info, epoch=None):
        super().log_info(info, epoch)
        for k, v in info.items():
            k = 'it_{}'.format(k)
            if k not in self.__further_it_keys:
                if k in self.history:
                    raise ValueError('{} is already part of the history.'.format(k))
                self.history[k] = DefaultList()
                self.__further_it_keys.append(k)
            self.history[k].append(v.data.item())

    def plot(self, suffix='.'):
        matplotlib.use('Agg')
        super().plot(suffix)
        outfile = pt.join(self.dir, suffix, 'it_err.pdf')
        plt.plot(self.history['it_err'], ls='-')
        legend = ["err"]
        if 'it_val_err' in self.__further_it_keys:
            plt.plot(self.history['it_val_err'], ls='-')
            legend += ['val_err']
        plt.legend(legend)
        plt.ylabel('error')
        plt.xlabel('it')
        plt.savefig(outfile, format='pdf')
        plt.close()
        for k in self.__further_it_keys:
            if k == 'it_val_err':
                continue
            plt.plot(self.history[k], ls='-')
            plt.ylabel(k.replace('it_', ''))
            plt.xlabel('it')
            plt.savefig(outfile.replace('it_err.pdf', k), format='pdf')
            plt.close()


class ProgressBar(object):
    EPOCH = datetime.utcfromtimestamp(0)

    def __init__(self, total, fps=2):
        self.total = total
        self.cur = 1
        self.time = (datetime.now() - ProgressBar.EPOCH).total_seconds()
        self.fps = fps

    def step(self):
        if (datetime.now() - ProgressBar.EPOCH).total_seconds() - self.time > 1/self.fps:
            percent = "{0:.2f}".format(100 * (self.cur / self.total))
            filled = int(60 * self.cur / self.total)
            bar = '-' * filled + ' ' * (60 - filled)
            print('\r|%s| %s%%' % (bar, percent), end='')
            self.time = (datetime.now() - ProgressBar.EPOCH).total_seconds()
        if self.cur == self.total:
            print()
        self.cur += 1

    def reset(self):
        self.cur = 1


def plot_many_roc(logdir, results, labels=None, name='roc', mean=False):
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

