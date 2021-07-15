#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   label_smoothing_cross_entropy_loss.py
Author:   zhanghao55@baidu.com
Date  :   21/01/12 14:05:51
Desc  :   
"""

import torch
import torch.nn as nn

class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, pred, label):
        num_class = pred.shape[1]
        pred =  self.softmax(pred)
        one_hot_label = nn.functional.one_hot(label, num_class).float()
        smoothed_one_hot_label = (1.0 - self.smoothing) * one_hot_label + self.smoothing / num_class
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=-1, keepdim=False)
        loss = loss.mean()
        return loss


if __name__ == "__main__":
    pass


