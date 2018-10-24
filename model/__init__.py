import copy
import torchvision.models as models
import torch

from models.fcn import *


def get_model(model_dict, n_classes, version=None):
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')


    if name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load("/home/pzl/pytorch-hed/model/vgg16.pth"))
        model.init_vgg16_params(vgg16)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
        }[name]
    except:
        raise("Model {} not available".format(name))
