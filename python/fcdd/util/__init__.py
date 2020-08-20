import json
from copy import deepcopy

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from torch import Tensor


class DefaultList(list):
    """ A list that automatically creates default entries if an index is accessed which is not yet set """
    def __init__(self, default=np.nan):
        list.__init__(self)
        self.__default = default

    def __setitem__(self, index, value):
        while len(self) <= index:
            self.append(deepcopy(self.__default))
        list.__setitem__(self, index, value)

    def __getitem__(self, item):
        if isinstance(item, slice):
            while len(self) <= (item.stop if item.stop is not None else item.start):
                self.append(deepcopy(self.__default))
        else:
            while len(self) <= item:
                self.append(deepcopy(self.__default))
        return list.__getitem__(self, item)


class CircleList(list):
    """
    A list that has a limited number of entries, which we term window size.
    An entry is set by using index modulo the window size.
    This works by the FIFO principle.
    """
    def __init__(self, window, default=np.nan):
        list.__init__(self)
        while len(self) < window:
            list.append(self, default)
        self.c = 0
        self.w = window

    def append(self, o):
        self.__setitem__(self.c % self.w, o)
        self.c += 1

    def __setitem__(self, index, value):
        list.__setitem__(self, index % self.w, value)
        self.c = index


class NumpyEncoder(json.JSONEncoder):
    """ Encoder to correctly use json on numpy arrays """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def imshow(tensors: Tensor, ms=100, name='out', nrow=8, norm=True, use_plt=True):
    """
    Given a tensor of images, this immediately displays them as a matrix of images using either matplotlib or opencv.
    :param tensors: tensor of images nxcxhxw
    :param ms: milliseconds to block while showing, only works for opencv
    :param name: name of the window that displays the images
    :param nrow: the number of images shown per row
    :param norm: whether to normalize the images to 0-1 range
    :param use_plt: whether to use matplotlib or opencv
    :return:
    """
    if use_plt:
        matplotlib.use('TkAgg')
    if isinstance(tensors, (list, tuple)):
        assert len(set([t.dim() for t in tensors])) == 1 and tensors[0].dim() == 4
        tensors = [t.float().div(255) if t.dtype == torch.uint8 else t for t in tensors]
        tensors = [t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t for t in tensors]
        tensors = torch.cat(tensors)
    if tensors.dtype == torch.uint8:
        tensors = tensors.float().div(255)
    t = vutils.make_grid(tensors, nrow=nrow, scale_each=norm)
    t = t.transpose(0, 2).transpose(0, 1).numpy()
    if use_plt:
        plt.close()
        if norm:
            plt.imshow(t, resample=True)
        else:
            plt.imshow(t, resample=True, vmin=0, vmax=1)
        plt.show()
        plt.pause(0.001)
    else:
        if t.shape[-1] == 3:
            t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        cv2.startWindowThread()
        cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(name, t)
        cv2.waitKey(ms)


def imsave(tensors: Tensor, path: str, nrow=8, norm=True, pad=2, padval=0):
    """
    Given a tensor of images, this encodes them and stores the images as a matrix of images at some destination.
    :param tensors: tensor of images nxcxhxw
    :param path: destination to store the images at
    :param nrow: the number images shown per row
    :param norm: whether to normalize the images to 0-1 range
    :param pad: the amount of padding used between the individual images
    :param padval: the pixel value that is used for padding
    :return:
    """
    if isinstance(tensors, (list, tuple)):
        assert len(set([t.dim() for t in tensors])) == 1 and tensors[0].dim() == 4
        tensors = [t.float().div(255) if t.dtype == torch.uint8 else t for t in tensors]
        tensors = [t.repeat(1, 3, 1, 1) if t.size(1) == 1 else t for t in tensors]
        tensors = torch.cat(tensors)
    if tensors.dtype == torch.uint8:
        tensors = tensors.float().div(255)
    t = vutils.make_grid(tensors, nrow=nrow, scale_each=norm, normalize=norm, padding=pad, pad_value=padval)
    t = t.transpose(0, 2).transpose(0, 1).numpy() * 255
    if t.shape[-1] == 3:
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, t)
