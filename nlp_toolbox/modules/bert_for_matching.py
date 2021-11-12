#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   bert_for_matching.py
Author:   zhanghao55@baidu.com
Date  :   21/07/15 18:53:00
Desc  :   
"""

import logging
import torch

from nlp_toolbox.modules.bert import BertPreTrainedModel, BertModel


class BertForMatching(BertPreTrainedModel):
    """用于匹配的bert
    """
    def __init__(self, config):
        super(BertForMatching, self).__init__(config)
        self.bert = BertModel(self.config)
        self.margin = self.config.other_config.get("margin", 0.5)
        #self.label_smooth_ratio = self.config.other_config.get("label_smooth_ratio", None)
        #logging.info("label_smooth_ratio: {}".format(self.label_smooth_ratio))
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-12)
        self.relu = torch.nn.ReLU()

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

    def forward(self, input_ids, second_input_ids=None, third_input_ids=None, labels=None, only_loss=False, **kwargs):
        """前向预测
        """
        # token_type_ids，position_ids 都在bert里自动造
        # 默认input_ids为一段话 position_ids即input_ids对应的索引
        # input_vec shape = [batch_size, self.config.pool_out_size]
        _, input_vec = self.bert(input_ids)

        res_dict = {
                "sent_vec": input_vec,
                }

        if second_input_ids is not None:
            # second_input_vec shape = [batch_size, self.config.pool_out_size]
            _, second_input_vec = self.bert(second_input_ids)
            res_dict["second_sent_vec"] = second_input_vec

            # 计算两输入间的相似度
            # second_sim shape = [batch_size]
            second_sim = self.cosine_sim(input_vec, second_input_vec)
            res_dict["second_sim"] = second_sim
            #logging.info("second_sim: {}".format(second_sim))

            if third_input_ids is not None:
                # third_input_vec shape = [batch_size, self.config.pool_out_size]
                _, third_input_vec = self.bert(third_input_ids)
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
                res_dict["loss"] = mean_loss
                #logging.info("triplet_loss: {}".format(triplet_loss))

            elif labels is not None:
                assert third_input_ids is None, "either third_input_ids or labels must be None"
                pointwise_loss = self.compute_pointwise_loss(second_sim, labels)
                res_dict["loss"] = pointwise_loss
        
        if only_loss:
            res_dict = {"loss": res_dict["loss"]}

        return res_dict
