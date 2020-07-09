from fcdd.models.shallow_cnn_28 import *
from fcdd.models.shallow_cnn_32 import *
from fcdd.models.shallow_cnn_224 import *
from fcdd.models.spatial_center_cnn_28 import *
from fcdd.models.spatial_center_cnn_32 import *
from fcdd.models.spatial_center_cnn_224 import *
import inspect
import sys


def choices():
    members = inspect.getmembers(sys.modules[__name__])
    clsses = [name for name, obj in members if inspect.isclass(obj)]
    return clsses


def all_nets():
    members = inspect.getmembers(sys.modules[__name__])
    clsses = {
        name:  ((obj, obj.pt_cls) if hasattr(obj, 'pt_cls') else (obj, None))
        for name, obj in members if inspect.isclass(obj)
    }
    return clsses


def load_nets(name, final_dim, in_shape, bias=False, dropout=None, **kwargs):
    """
    Loads the nets.

    :param dim: final output dimension, i.e. embedding size
    :param chin: input shape c x h x w
    """

    implemented_nets = choices()
    assert len(implemented_nets) == len(set(implemented_nets))
    assert name in implemented_nets

    NET, PT_NET = all_nets()[name]
    net = NET(final_dim, in_shape, bias=bias, dropout=dropout, **kwargs)
    pt_net = PT_NET(net, dropout=dropout) if PT_NET is not None else None

    return net, pt_net

