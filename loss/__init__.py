#-*-coding:utf-8-*-
'''
Created on Oct 24,2018

@author: pengzhiliang
'''
import copy
import functools

from loss import cross_entropy2d
# from loss import bootstrapped_cross_entropy2d
# from loss import multi_scale_cross_entropy2d



key2loss = {'cross_entropy': cross_entropy2d,
            }

def get_loss_function(cfg):
    if cfg['training']['loss'] is None:
        print("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        print('Using {} with {} params'.format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
