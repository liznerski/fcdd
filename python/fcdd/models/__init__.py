import inspect
import sys
from typing import List, Dict, Tuple

from fcdd.models.shallow_cnn_28 import *
from fcdd.models.shallow_cnn_32 import *
from fcdd.models.shallow_cnn_224 import *
from fcdd.models.fcdd_cnn_28 import *
from fcdd.models.fcdd_cnn_32 import *
from fcdd.models.fcdd_cnn_224 import *


def choices() -> List[str]:
    """ returns all model names """
    members = inspect.getmembers(sys.modules[__name__])
    clsses = [name for name, obj in members if inspect.isclass(obj)]
    return clsses


def all_nets() -> Dict[str, Tuple[torch.nn.Module, type]]:
    """ returns a mapping form model names to a tuple of (model instance, corresponding AE class) """
    members = inspect.getmembers(sys.modules[__name__])
    clsses = {
        name:  ((obj, obj.encoder_cls) if hasattr(obj, 'encoder_cls') else (obj, None))
        for name, obj in members if inspect.isclass(obj)
    }
    return clsses


def load_nets(name: str, in_shape: List[int], bias=False, **kwargs) -> torch.nn.Module:
    """
    Creates an instance of a network architecture.
    :param name: name of the model of which an instance is to be created.
    :param in_shape: shape of the inputs the model expects (n x c x h x w).
    :param bias: whether to use bias in the model.
    :param kwargs: further specific parameters. See network architectures.
    :return: (instance of model, instance of AE version of the model)
    """
    implemented_nets = choices()
    assert len(implemented_nets) == len(set(implemented_nets)), 'some model has not a unique name!'
    assert name in implemented_nets, 'name {} is not a known model!'.format(name)

    NET, ENCODER_NET = all_nets()[name]
    if ENCODER_NET is not None:
        net = NET(ENCODER_NET(in_shape, bias=bias, **kwargs))
    else:
        net = NET(in_shape, bias=bias, **kwargs)

    return net

