from abc import abstractmethod, ABC
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from fcdd.datasets.noise import gkern


def ceil(x: float):
    return int(np.ceil(x))


class ReceptiveModule(torch.nn.Module, ABC):
    """ Baseclass for network modules that provide an upsampling based on the receptive field using Gaussian kernels """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._n = None  # feature length
        self._r = None  # receptive field extent
        self._j = None  # receptive field jump
        self._s = None  # receptive field shift
        self._in_shape = None  # input shape (n x c x h x w)

    @property
    def reception(self):
        """
        Returns receptive field information, i.e.
        {'n': feature length, 'j' jump, 'r' extent, 's' shift, 'i' input shape}.
        """
        return {'n': self._n, 'j': self._j, 'r': self._r, 's': self._s, 'img_shape': self._in_shape}

    def set_reception(self, n: int, j: int, r: float, s: float, in_shape: List[int] = None):
        self._n = n
        self._j = j
        self._r = r
        self._s = s
        if in_shape is not None:
            self._in_shape = in_shape

    def receptive_upsample(self, pixels: torch.Tensor, reception=True, std: float = None, cpu=True) -> torch.Tensor:
        """
        Implement this to upsample given tensor images based on the receptive field with a Gaussian kernel.
        Usually one can just invoke the receptive_upsample method of the last convolutional layer.
        :param pixels: tensors that are to be upsampled (n x c x h x w)
        :param reception: whether to use reception. If 'False' uses nearest neighbor upsampling instead.
        :param std: standard deviation of Gaussian kernel. Defaults to kernel_size_to_std in fcdd.datasets.noise.py.
        :param cpu: whether the output should be on cpu or gpu
        """
        if self.reception is None or any([i not in self.reception for i in ['j', 's', 'r', 'img_shape']]) \
                or not reception:
            if reception:
                self.logger.logtxt(
                    'Fell back on nearest neighbor upsampling since reception was not available!', print=True
                )
            return self.__upsample_nn(pixels)
        else:
            assert pixels.dim() == 4 and pixels.size(1) == 1, 'receptive upsample works atm only for one channel'
            pixels = pixels.squeeze(1)
            if self.reception is None:
                raise ValueError('receptive field is unknown!')
            ishape = self.reception['img_shape']
            pixshp = pixels.shape
            # regarding s: if between pixels, pick the first
            s, j, r = int(self.reception['s']), self.reception['j'], self.reception['r']
            gaus = torch.from_numpy(gkern(r, std)).float().to(pixels.device)
            pad = (r - 1) // 2
            if (r - 1) % 2 == 0:
                res = torch.nn.functional.conv_transpose2d(
                    pixels.unsqueeze(1), gaus.unsqueeze(0).unsqueeze(0), stride=j, padding=0,
                    output_padding=ishape[-1] - (pixshp[-1] - 1) * j - 1
                )
            else:
                res = torch.nn.functional.conv_transpose2d(
                    pixels.unsqueeze(1), gaus.unsqueeze(0).unsqueeze(0), stride=j, padding=0,
                    output_padding=ishape[-1] - (pixshp[-1] - 1) * j - 1 - 1
                )
            out = res[:, :, pad - s:-pad - s, pad - s:-pad - s]  # shift by receptive center (s)
            return out if not cpu else out.cpu()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def __upsample_nn(self, pixels):
        res = torch.nn.functional.interpolate(pixels, self.reception['img_shape'][1:])
        return res


class RecConv2d(torch.nn.Conv2d, ReceptiveModule):
    """
    Like torch.nn.Conv2d, but sets its own receptive field information based on the receptive field information
    of the previous layer:
    :param in_width: the width = height of the output
    :param in_jump: the distance between two adjacent features in this layer's output
        (or jump) w.r.t. to the overall network input. For instance, for j=2 the centers of the receptive field
        of two adjacent pixels in this layer's output have a distance of 2 pixels.
    :param in_reception: the receptive field extent r
    :param in_start: the shift of the receptive field

    cf. https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 in_width, in_jump, in_reception, in_start, img_shape,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.set_reception(
            (in_width + 2 * padding - kernel_size) // stride + 1,
            in_jump * stride,
            in_reception + (kernel_size - 1) * in_jump,
            in_start + ((kernel_size - 1) / 2 - padding) * in_jump,
            img_shape
        )

    @property
    def reception(self):
        return super().reception


class RecMaxPool2d(torch.nn.MaxPool2d, ReceptiveModule):
    """
    Like torch.nn.MaxPool2d, but sets its own receptive field information based on the receptive field information
    of the previous layer:
    :param in_width: the width = height of the output of layer
    :param in_jump: the distance between two adjacent features in this layer's output
        (or jump) w.r.t. to the overall network input. For instance, for j=2 the centers of the receptive field
        of two adjacent pixels in this layer's output have a distance of 2 pixels.
    :param in_reception: the receptive field extent r
    :param in_start: the shift of the receptive field

    cf. https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    """
    def __init__(self, kernel_size, in_width, in_jump, in_reception, in_start, img_shape,
                 stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.set_reception(
            (in_width + 2 * padding - kernel_size) // stride + 1,
            in_jump * stride,
            in_reception + (kernel_size - 1) * in_jump,
            in_start + ((kernel_size - 1) / 2 - padding) * in_jump,
            img_shape
        )

    @property
    def reception(self):
        return super().reception


class BaseNet(torch.nn.Module, ABC):
    """ Base class for all networks """

    def __init__(self, in_shape: Tuple[int, int, int], bias=False, **kwargs):
        """
        :param in_shape: the shape the model expects the input to have (n x c x h x w).
        :param bias: whether to use bias in the network layers.
        :param kwargs: further specific parameters. See network architectures.
        """
        super().__init__()
        assert len(in_shape) == 3 and in_shape[1] == in_shape[2]
        self._in_shape = in_shape
        self.__bias = bias

    @property
    def bias(self):
        return self.__bias

    @property
    def in_shape(self):
        return self._in_shape

    def get_grad_heatmap(self, losses: torch.Tensor, inputs: torch.Tensor, method='grad', absolute=True) -> torch.tensor:
        """
        Compute gradient heatmaps of loss w.r.t. to inputs.
        :param losses: the computed loss of some training iteration for this model.
        :param input: the inputs that have been used for losses and outputs (n x c x h x w).
        :param method: whether to return heatmaps based on the pure gradients ('grad') or
            use the gradients to weight the inputs ('xgrad').
        :param absolute: whether to take the absolute value as a last step in the computation.
        """
        methods = ('grad', 'xgrad')
        assert method in methods
        grads = torch.autograd.grad((*losses.view(losses.size(0), -1).mean(-1),), inputs, create_graph=True)[0]
        if method == 'xgrad':
            heatmaps = inputs.detach() * grads
        else:
            heatmaps = grads
        if absolute:
            heatmaps = heatmaps.abs()
        heatmaps = heatmaps.sum(1, keepdim=True)
        # heatmaps /= heatmaps.sum((2, 3), keepdim=True)
        return heatmaps.detach()


class ReceptiveNet(BaseNet, ReceptiveModule):
    def __init__(self, in_shape: Tuple[int, int, int], bias=False, **kwargs):
        """
        Base class for neural networks that keep track of the receptive field flow, i.e.
        the receptive field (extent, shift, jump, etc.) can be retrieved at any time via the according property.
        To be able to keep track, all layers that change the receptive field must be created via
        the class' methods, i.e._create_conv2d and _create_maxpool2d.

        :param in_shape: the shape the model expects the input to have (n x c x h x w).
        :param bias: whether to use bias in the network layers.
        :param kwargs: further specific parameters. See network architectures.
        """
        super().__init__(in_shape, bias, **kwargs)
        self.set_reception(in_shape[1], 1, 1, 0)
        self.__initial_reception = deepcopy(self.reception)

    @property
    def reception(self):
        return super().reception

    @property
    def initial_reception(self):
        return self.__initial_reception

    def reset_parameters(self):
        self.apply(self.__weight_reset)

    def __weight_reset(self, m):
        if m == self:
            return
        try:
            m.reset_parameters()
        except AttributeError as e:
            if len(list(m.parameters())) > 0:
                if isinstance(m, (ReceptiveNet, ReceptiveModule)):
                    pass
                else:
                    raise e
            else:
                pass

    def _create_conv2d(self, in_channels: int, out_channels: int, kernel_size: int,
                       stride=1, padding=0, dilation=1, groups=1,
                       bias=True, padding_mode='zeros') -> RecConv2d:
        """
        Creates a convolutional layer with receptive field information based on the current receptive field of
        the overall model.
        WARNING:
        Using this method does only work if all layers are created with create-methods like this one.
        If layers that change the receptive field (e.g. fully connected layers, certain attention layers)
        are manually appended, the model is not informed about the change of receptive field and
        thus further layers created by this method will have false receptive field information.
        Also, of cause, layers must be used in the order in which they have been created, and must be used exactly once.

        :param in_channels: number of channels in the input image.
        :param out_channels: number of channels produced by the convolution-
        :param kernel_size: size of the convolving kernel
        :param stride: stride of the convolution.
        :param padding: zero-padding added to both sides of the input.
        :param dilation: spacing between kernel elements.
        :param groups: number of blocked connection from input channels to output channels.
        :param bias: whether to use a bias in the layer.
        :param padding_mode: accepted values 'zeros' and 'circular'.
        :return: convolutional layer
        """
        rec = self.reception
        n, j, r, s, in_shape = rec['n'], rec['j'], rec['r'], rec['s'], rec['img_shape']
        conv = RecConv2d(
            in_channels, out_channels, kernel_size, n, j, r, s, in_shape,
            stride, padding, dilation, groups, bias, padding_mode
        )
        rec = conv.reception
        self.set_reception(rec['n'], rec['j'], rec['r'], rec['s'], rec['img_shape'])
        return conv

    def _create_maxpool2d(self, kernel_size: int, stride: int = None, padding=0, dilation=1,
                          return_indices=False, ceil_mode=False) -> RecMaxPool2d:
        """
        Creates a pool layer with receptive field information based on the current receptive field of
        the overall model.
        WARNING:
        Using this method does only work if all layers are created with create-methods like this one.
        If layers that change the receptive field (e.g. fully connected layers, certain attention layers)
        are manually appended, the model is not informed about the change of receptive field and
        thus further layers created by this method will have false receptive field information.
        Also, of cause, layers must be used in the order in which they have been created, and must be used exactly once.

        :param kernel_size: the size of the window to take a max over.
        :param stride: the stride of the window. Default value is kernel_size.
        :param padding: implicit zero padding to be added on both sides.
        :param dilation: a parameter that controls the stride of elements in the window.
        :param return_indices: whether to return the max indices along with the outputs.
        :param ceil_mode: whether to use `ceil` instead of `floor` to compute the output shape
        :return: max pool layer
        """
        rec = self.reception
        n, j, r, s, in_shape = rec['n'], rec['j'], rec['r'], rec['s'], rec['img_shape']
        pool = RecMaxPool2d(
            kernel_size, n, j, r, s, in_shape,
            stride, padding, dilation, return_indices, ceil_mode
        )
        rec = pool.reception
        self.set_reception(rec['n'], rec['j'], rec['r'], rec['s'], rec['img_shape'])
        return pool


class FCDDNet(ReceptiveNet):
    """ Baseclass for FCDD networks, i.e. network without fully connected layers that have a spatial output """
    def __init__(self, in_shape: Tuple[int, int, int], bias=False, **kwargs):
        super().__init__(in_shape, bias)



