#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   textcnn.py
Author:   zhanghao55@baidu.com
Date  :   21/01/14 10:53:03
Desc  :   
"""

import sys
import logging
import torch
import torch.nn as nn

from nlp_toolbox.modules.basic_layers import EmbeddingLayer, TextCNNLayer
from nlp_toolbox.modules.losses.label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropyLoss


class TextCNNClassifier(nn.Module):
    """textcnn分类模型
    """
    def __init__(self,
            num_class,
            vocab_size,
            emb_dim=32,
            num_filters=10,
            fc_hid_dim=32,
            num_channels=1,
            win_size_list=None,
            emb_sparse=True,
            label_smooth_ratio=None
            ):
        super(TextCNNClassifier, self).__init__()

        self.num_class = num_class
        self.label_smooth_ratio = label_smooth_ratio

        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            is_sparse=emb_sparse,
            )

        if win_size_list is None:
            win_size_list = [3]

        self.textcnn = TextCNNLayer(
            emb_dim,
            num_filters,
            num_channels,
            win_size_list,
            fc_hid_dim,
            )

        logging.info("num_class     = {}".format(num_class))
        logging.info("vocab size    = {}".format(vocab_size))
        logging.info("emb_dim       = {}".format(emb_dim))
        logging.info("num filters   = {}".format(num_filters))
        logging.info("fc_hid_dim    = {}".format(fc_hid_dim))
        logging.info("num channels  = {}".format(num_channels))
        logging.info("win size list = {}".format(win_size_list))
        logging.info("is sparse     = {}".format(emb_sparse))

        #self._hid_fc = nn.Linear(in_features=num_filters * len(win_size_list), out_features=fc_hid_dim)

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
        #print("\n".join(map(lambda ids: "/ ".join([id_2_token[x] for x in ids]), inputs.numpy())))
        # inputs shape = [batch_size, seq_len]
        #print("inputs shape: {}".format(inputs.shape))

        # emb shape = [batch_size, seq_len, emb_dim]
        emb = self.embedding(input_ids)
        #print("emb shape: {}".format(emb.shape))

        conv_pool_res = self.textcnn(emb)

        #hid_fc = self._hid_fc(conv_pool_res)
        #print("hid_fc shape: {}".format(hid_fc.shape))

        hid_act = self._act(conv_pool_res)

        logits = self._output_fc(hid_act)
        #print("logits shape: {}".format(logits.shape))

        softmax_pred = self.softmax(logits)

        res_dict = {
                "sent_logits": logits,
                "sent_softmax": softmax_pred,
                }

        # 如果没有给标签 则输出logits结果
        if labels is not None:
            loss = self.compute_loss(softmax_pred, labels)
            res_dict["loss"] = loss

        ## 调整label的形状
        #if len(labels.shape) == 1:
        #    labels = torch.reshape(labels, shape=(-1,))
        ##logging.info("labels shape: {}".format(labels.shape))
        ##logging.info("logits shape: {}".format(logits.shape))

        #loss = self.ce_loss(logits, labels)

        #res_dict["loss"] = loss

        return res_dict

if __name__ == "__main__":
    model_config = {
        "num_class": 12,
        "vocab_size": 3000,
        "emb_dim": 32,
        "num_filters": 10,
        "fc_hid_dim": 64,
        "num_channels": 1,
        "win_size_list": None,
        "emb_sparse": True,
    }
    TextCNNClassifier(**model_config)
