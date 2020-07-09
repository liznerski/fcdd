from abc import abstractmethod

import numpy as np
import torch
from fcdd.datasets.noise import gkern


def ceil(x):
    return int(np.ceil(x))


class ReceptiveNet(torch.nn.Module):
    def __init__(self, final_dim, in_shape, bias=False, **kwargs):
        super().__init__()
        assert len(in_shape) == 3 and in_shape[1] == in_shape[2]
        self.__in_shape = in_shape
        self.__final_dim = final_dim
        self.__bias = bias
        self.__n = in_shape[2]
        self.__r = 1
        self.__j = 1
        self.__s = 0
        self.__initial_reception = {
            'n': self.__n, 'j': self.__j, 'r': self.__r, 's': self.__s, 'img_shape': self.__in_shape
        }

    @property
    def reception(self):
        return {'n': self.__n, 'j': self.__j, 'r': self.__r, 's': self.__s, 'img_shape': self.in_shape}

    def set_reception(self, n, j, r, s):
        self.__n = n
        self.__j = j
        self.__r = r
        self.__s = s

    @property
    def bias(self):
        return self.__bias

    @property
    def in_shape(self):
        return self.__in_shape

    @property
    def final_dim(self):
        return self.__final_dim

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

    def _create_conv2d(self, in_channels, out_channels, kernel_size,
                       stride=1, padding=0, dilation=1, groups=1,
                       bias=True, padding_mode='zeros'):
        """
        Creates a conv layer with receptional field information based on the current receptional field.
        WARNING: using this method does only work if all layers are created with such create-methods.
        If layers, that change the receptive field (e.g. fully connected layers, certain attention layers),
        are manually appended, the class is not informed about the change of receptive field and
        thus further layer created by this method will have false receptive field information.
        Also, of cause, layers must be used in the order in which they have been created, and must be used exactly once.
        """
        conv = RecConv2d(
            in_channels, out_channels, kernel_size, self.__n, self.__j, self.__r, self.__s, self.__in_shape,
            stride, padding, dilation, groups, bias, padding_mode
        )
        self.__n, self.__j, self.__r, self.__s = conv.n, conv.j, conv.r, conv.start
        return conv

    def _create_maxpool2d(self, kernel_size, stride=None, padding=0, dilation=1,
                          return_indices=False, ceil_mode=False):
        """
        Creates a pool layer with receptional field information based on the current receptional field.
        WARNING: using this method does only work if all layers are created with such create-methods.
        If layers, that change the receptive field (e.g. fully connected layers, certain attention layers),
        are manually appended, the class is not informed about the change of receptive field and
        thus further layer created by this method will have false receptive field information.
        Also, of cause, layers must be used in the order in which they have been created, and must be used exactly once.
        """
        pool = RecMaxPool2d(
            kernel_size, self.__n, self.__j, self.__r, self.__s, self.__in_shape,
            stride, padding, dilation, return_indices, ceil_mode
        )
        self.__n, self.__j, self.__r, self.__s = pool.n, pool.j, pool.r, pool.start
        return pool

    @abstractmethod
    def get_grad_heatmap(self, losses: torch.Tensor, outputs: torch.Tensor, inputs: torch.Tensor,
                         method='grad', absolute=True):
        return None


class SpatialCenterNet(ReceptiveNet):
    def __init__(self, final_dim, in_shape, bias=False, **kwargs):
        super().__init__(final_dim, in_shape, bias)

    @abstractmethod
    def get_heatmap(self, outs, threshold=False, invert=False, reception=True):
        return None

    def get_grad_heatmap(self, losses: torch.Tensor, outputs: torch.Tensor, inputs: torch.Tensor,
                         method='grad', absolute=True):
        """
        Compute (input x) gradient (saliency) heatmaps of inputs (B x C x H x W) for given network outputs.
        """
        methods = ('grad', 'xgrad')
        assert method in methods
        grads = torch.autograd.grad((*losses.view(losses.size(0), -1).mean(-1), ), inputs, create_graph=True)[0]
        if method == 'xgrad':
            heatmaps = inputs.detach() * grads
        else:
            heatmaps = grads
        if absolute:
            heatmaps = heatmaps.abs()
        heatmaps = heatmaps.sum(1, keepdim=True)
        # heatmaps /= heatmaps.sum((2, 3), keepdim=True)
        return heatmaps.detach()


class ReceptiveModule(torch.nn.Module):
    @property
    @abstractmethod
    def reception(self):
        pass

    @abstractmethod
    def get_heatmap(self, pixels, heads=10, threshold=False, invert=False, reception=True):
        return None

    @property
    def device(self):
        return list(self.parameters())[0].device

    def _get_single_head_heatmap(self, pixels, reception=True, std=None, cpu=True):
        if self.reception is None or any([i not in self.reception for i in ['j', 's', 'r', 'img_shape']]) \
                or not reception:
            if reception:
                self.logger.logtxt(
                    'Fell back on upsample method for heatmap since reception was not available!', print=True
                )
            return self.__upsample_map(pixels)
        else:
            cap = 4
            s = self.__max_memory(pixels)
            if s > cap:
                outs = []
                pixel_splitss = pixels.split(pixels.size(0) // ceil(self.__max_memory(pixels) / cap))
                print(
                    'Had to split Receptive Field Upsampling in {} tiles of {} images, '
                    'because overall it takes {} GB for all {} images..'
                    .format(len(pixel_splitss), pixel_splitss[0].size(0), s, pixels.size(0))
                )
                for n, pixels_split in enumerate(pixel_splitss):
                    print('Receptive Field Upsampling: Processing tile {}/{}...'.format(n, len(pixel_splitss)))
                    assert self.__max_memory(pixels_split) <= cap
                    outs.append(self._receptive_field_map(pixels_split.to(self.device), std))
                    if cpu:
                        outs[-1] = outs[-1].cpu()
                return torch.cat(outs, dim=1)
            else:
                out = self._receptive_field_map(pixels.to(self.device), std)
                return out if not cpu else out.cpu()

    def __max_memory(self, pixels, elemsize=4):
        # elemsize is the byte size of one element of the tensor, for float32 this is 4
        assert pixels.dim() == 3  # This considers only one head, i.e. n x ah x aw shape
        ishape = self.reception['img_shape']
        r = self.reception['r']
        s = np.prod((elemsize, pixels.size(0), pixels.size(1), *(i + r // 2 for i in ishape[2:])))  # byte
        s = s / 1024**3  # gigabyte
        return s

    def __upsample_map(self, attention_weights, threshold=False, invert=False):
        assert attention_weights.dim() == 3  # n x ah x aw
        attention_weights = self.transform_attentions(attention_weights, threshold, invert)
        res = torch.nn.functional.interpolate(attention_weights[:, None, :, :], self.reception['img_shape'][1:])
        if self.reception['img_shape'][0] == 3:
            res = res.repeat(1, 3, 1, 1)  # repeat to 3 channels
        return res[None, :, :, :, :]  # heads x n x c x h x w

    def _receptive_field_map(self, pixels, std=None):
        assert pixels.dim() == 3  # This considers only one head, i.e. n x ah x aw shape
        if self.reception is None:
            raise ValueError('receptive field is unknown!')
        ishape = self.reception['img_shape']
        pixshp = pixels.shape
        # regarding s: if between pixels, pick the first
        s, j, r = int(self.reception['s']), self.reception['j'], self.reception['r']
        gaus = torch.from_numpy(gkern(r, std)).float().to(pixels.device)
        pad = (r-1)//2
        if (r-1) % 2 == 0:
            res = torch.nn.functional.conv_transpose2d(
                pixels.unsqueeze(1), gaus.unsqueeze(0).unsqueeze(0), stride=j, padding=0,
                output_padding=ishape[-1] - (pixshp[-1] - 1) * j - 1
            ).unsqueeze(0)
        else:
            res = torch.nn.functional.conv_transpose2d(
                pixels.unsqueeze(1), gaus.unsqueeze(0).unsqueeze(0), stride=j, padding=0,
                output_padding=ishape[-1] - (pixshp[-1] - 1) * j - 1 - 1
            ).unsqueeze(0)
        return res[:, :, :, pad-s:-pad-s, pad-s:-pad-s]  # shift by receptive center (s)


class RecConv2d(torch.nn.Conv2d, ReceptiveModule):
    """
    Like super, but keeps track of the receptive field center position and size by
    processing the receptive field parameters of the previous conv layer:
    :param in_width: the number of 1D features n, i.e. width = height of the input
    :param in_jump: the distance between two adjacent features (or jump) j
    :param in_reception: the receptive field size r
    :param in_start: the center coordinate of the upper left feature (the first feature) start

    cf. https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 in_width, in_jump, in_reception, in_start, img_shape,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.n = (in_width + 2 * padding - kernel_size) // stride + 1
        self.j = in_jump * stride
        self.r = in_reception + (kernel_size - 1) * in_jump
        self.start = in_start + ((kernel_size - 1) / 2 - padding) * in_jump
        self.img_shape = img_shape

    @property
    def reception(self):
        return {'n': self.n, 'j': self.j, 'r': self.r, 's': self.start, 'img_shape': self.img_shape}

    def get_heatmap(self, pixel_heads, heads=10, threshold=False, invert=False, reception=True, std=None, cpu=True):
        # NOTE THAT PIXELS HERE ARE NOT ATTENTION_WEIGHTS BUT THE ACTUAL OUTPUT!
        if pixel_heads.dim() != 4:  # most likely attention weights have been inputted
            return None
        else:
            n, iheads, ih, iw = pixel_heads.shape
            heatmap = []
            pixel_heads = pixel_heads.transpose(0, 1)[:heads]
            for pixels in pixel_heads:
                heatmap.append(self._get_single_head_heatmap(pixels, reception, std=std, cpu=cpu))
            return torch.cat(heatmap).transpose(0, 1)


class RecMaxPool2d(torch.nn.MaxPool2d, ReceptiveModule):
    """
    Like super, but keeps track of the receptive field center position and size by
    processing the receptive field parameters of the previous conv layer:
    :param in_width: the number of 1D features n, i.e. width = height of the input
    :param in_jump: the distance between two adjacent features (or jump) j
    :param in_reception: the receptive field size r
    :param in_start: the center coordinate of the upper left feature (the first feature) start

    cf. https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    """
    def __init__(self, kernel_size, in_width, in_jump, in_reception, in_start, img_shape,
                 stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.n = (in_width + 2 * padding - kernel_size) // stride + 1
        self.j = in_jump * stride
        self.r = in_reception + (kernel_size - 1) * in_jump
        self.start = in_start + ((kernel_size - 1) / 2 - padding) * in_jump
        self.img_shape = img_shape

    @property
    def reception(self):
        return {'n': self.n, 'j': self.j, 'r': self.r, 's': self.start, 'img_shape': self.img_shape}

    def get_heatmap(self, pixels, heads=10, threshold=False, invert=False, reception=True, cpu=True):
        attention_weights = pixels.mean(1)  # mean over all channels
        return self._get_single_head_heatmap(attention_weights, threshold, cpu=cpu)

