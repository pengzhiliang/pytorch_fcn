#-*-coding:utf-8-*-
'''
Created on Oct 24,2018

@author: pengzhiliang
'''
import torch.nn.functional as F
import numpy as np


def cross_entropy2d(input, target, weight=None, reduction='elementwise_mean'):
    """
    输入输出图片与mask的交叉熵损失�?d平面�?    """
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction='elementwise_mean', ignore_index=250
    )
    return loss