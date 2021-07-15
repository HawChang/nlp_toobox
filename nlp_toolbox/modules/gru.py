#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   gru.py
Author:   zhanghao55@baidu.com
Date  :   21/01/14 16:23:02
Desc  :   
"""

import sys
import logging
import torch
import torch.nn as nn

from nlp_toolbox.modules.basic_layers import EmbeddingLayer
from nlp_toolbox.modules.losses.label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropyLoss


class GRUClassifier(nn.Module):
    """GRU分类模型
    """
    def __init__(self,
            num_class,
            vocab_size,
            emb_dim=128,
            gru_dim=256,
            gru_layers=1,
            fc_hid_dim=256,
            emb_sparse=True,
            bidirection=True,
            label_smooth_ratio=None,
            ):
        super(GRUClassifier, self).__init__()

        logging.info("num_class   = {}".format(num_class))
        logging.info("vocab_size  = {}".format(vocab_size))
        logging.info("emb_dim     = {}".format(emb_dim))
        logging.info("gru_dim     = {}".format(gru_dim))
        logging.info("gru_layers  = {}".format(gru_layers))
        logging.info("fc_hid_dim  = {}".format(fc_hid_dim))
        logging.info("emb_sparse  = {}".format(emb_sparse))
        logging.info("bidirection = {}".format(bidirection))

        #self.bidirection = bidirection
        self.num_class = num_class
        self.label_smooth_ratio = None

        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            is_sparse=emb_sparse)

        self.gru = nn.GRU(
                input_size=emb_dim,
                hidden_size=gru_dim,
                num_layers=gru_layers,
                batch_first=True,
                bidirectional=bidirection)

        if bidirection:
            self._hid_fc2 = nn.Linear(in_features=gru_dim * 2, out_features=fc_hid_dim)
        else:
            self._hid_fc2 = nn.Linear(in_features=gru_dim, out_features=fc_hid_dim)

        self._act = nn.Tanh()

        self._output_fc = nn.Linear(in_features=fc_hid_dim, out_features=num_class)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.ce_loss = nn.CrossEntropyLoss()

    def compute_loss(self, softmax_pred, labels):
        """
        计算loss
        softmax_pred: (batch_size, 1)
        """
        softmax_pred = softmax_pred.view(-1, self.num_class)
        labels = labels.view(-1)
        if self.label_smooth_ratio is not None:
            loss = LabelSmoothingCrossEntropyLoss(self.label_smooth_ratio)(softmax_pred, labels)
        else:
            # CrossEntropyLoss = Softmax + log + NLLLoss
            loss = torch.nn.NLLLoss(reduction="mean")(torch.log(softmax_pred), labels)

        return loss

    def forward(self, input_ids, *args, labels=None, **kwargs):
        """前向预测
        """
        self.gru.flatten_parameters()

        input_emb = self.embedding(input_ids)
        #logging.info("input_emb shape: {}".format(input_emb.shape))

        gru_out, _ = self.gru(input_emb)
        #logging.info("gru_out shape: {}".format(gru_out.shape))

        #encoded_vector = L.reduce_max(encoded_vector, dim=1)
        # 取最后一个时间步的输出
        gru_out = gru_out[:,-1]
        #logging.info("gru_out last output  shape: {}".format(gru_out.shape))

        hidden_out = self._hid_fc2(gru_out)
        #logging.info("hidden out shape: {}".format(hidden_out.shape))

        hidden_out = self._act(hidden_out)

        logits = self._output_fc(hidden_out)
        #logging.info("logits shape: {}".format(logits.shape))

        softmax_pred = self.softmax(logits)

        res_dict = {
                "sent_logits": logits,
                "sent_softmax": softmax_pred,
                }

        # 如果没有给标签 则输出logits结果
        if labels is not None:
            loss = self.compute_loss(softmax_pred, labels)
            res_dict["loss"] = loss

        #if len(labels.shape) == 1:
        #    labels = torch.reshape(labels, shape=(-1,))
        ##print("labels shape: {}".format(labels.shape))

        #loss = self.ce_loss(logits, labels)
        ## 如果输出logits的激活函数为softmax 则不能用softmax_with_cross_entropy
        ##loss = L.cross_entropy(logits, labels)
        ##loss = L.reduce_mean(loss)
        #res_dict["loss"] = loss
        return res_dict


if __name__ == "__main__":
    pass


