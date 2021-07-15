#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   textcnn_matching.py
Author:   zhanghao55@baidu.com
Date  :   21/07/13 17:53:03
Desc  :   
"""

import sys
import logging
import torch
import torch.nn as nn

from nlp_toolbox.modules.basic_layers import EmbeddingLayer, TextCNNLayer
from nlp_toolbox.modules.losses.label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropyLoss


class TextCNNMatching(nn.Module):
    """textcnn匹配模型
    """
    def __init__(self,
            vocab_size,
            emb_dim=32,
            num_filters=128,
            num_channels=1,
            win_size_list=None,
            output_dim=128,
            emb_sparse=True,
            label_smooth_ratio=None,
            margin=0.3,
            ):
        super(TextCNNMatching, self).__init__()

        self.label_smooth_ratio = label_smooth_ratio
        self.margin = margin

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
            output_dim,
            )

        self.cosine_sim = nn.CosineSimilarity(dim=-1, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2.0, reduction="mean")

        logging.info("vocab size    = {}".format(vocab_size))
        logging.info("emb_dim       = {}".format(emb_dim))
        logging.info("num filters   = {}".format(num_filters))
        logging.info("num channels  = {}".format(num_channels))
        logging.info("win size list = {}".format(win_size_list))
        logging.info("output_dim    = {}".format(output_dim))
        logging.info("is sparse     = {}".format(emb_sparse))
        logging.info("margin        = {}".format(margin))

    def compute_pointwise_loss(self, tar_sim, labels):
        """
        计算loss
        """
        return torch.sqrt((tar_sim - labels).pow(2).mean())

    def compute_triplet_loss(self, pos_sim, neg_sim):
        """
        计算loss
        """
        #logging.info("origin diff: {}".format(pos_sim - neg_sim + self.margin))
        return self.relu(neg_sim - pos_sim + self.margin)

    def forward(self, input_ids, second_input_ids=None, third_input_ids=None, labels=None, **kwargs):
        """前向预测
        """
        #print("\n".join(map(lambda ids: "/ ".join([id_2_token[x] for x in ids]), inputs.numpy())))
        # inputs shape = [batch_size, seq_len]
        #print("inputs shape: {}".format(inputs.shape))

        # emb shape = [batch_size, seq_len, emb_dim]
        input_emb = self.embedding(input_ids)
        #print("emb shape: {}".format(emb.shape))

        # input_vec shape = [batch_size, output_dim]
        input_vec = self.textcnn(input_emb)

        res_dict = {
                "sent_vec": input_vec,
                }

        if second_input_ids is not None:
            second_input_emb = self.embedding(second_input_ids)
            # second_input_vec shape = [batch_size, output_dim]
            second_input_vec = self.textcnn(second_input_emb)
            res_dict["second_sent_vec"] = second_input_vec

            # 计算两输入间的相似度
            # second_sim shape = [batch_size]
            second_sim = self.cosine_sim(input_vec, second_input_vec)
            res_dict["second_sim"] = second_sim
            #logging.info("second_sim: {}".format(second_sim))

            if third_input_ids is not None:
                third_input_emb = self.embedding(third_input_ids)
                # third_input_vec shape = [batch_size, output_dim]
                third_input_vec = self.textcnn(third_input_emb)
                res_dict["third_sent_vec"] = third_input_vec

                # 计算两输入间的相似度
                # third_sim shape = [batch_size]
                third_sim = self.cosine_sim(input_vec, third_input_vec)
                res_dict["third_sim"] = third_sim
                #logging.info("third_sim: {}".format(third_sim))

                #triplet_loss = self.triplet_loss(input_vec, second_input_vec, third_input_vec)
                triplet_loss = self.compute_triplet_loss(second_sim, third_sim)
                res_dict["each_loss"] = triplet_loss

                mean_loss = triplet_loss.mean()
                res_dict["loss"] = triplet_loss
                #logging.info("triplet_loss: {}".format(triplet_loss))

            elif labels is not None:
                assert third_input_ids is None, "either third_input_ids or labels must be None"
                pointwise_loss = self.compute_pointwise_loss(second_sim, labels)
                res_dict["loss"] = pointwise_loss

        return res_dict
