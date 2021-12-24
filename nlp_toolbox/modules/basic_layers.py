#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   basic_layers.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/01/14 16:14:58
Desc  :   
"""

import sys
import logging
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Embedding Layer class
    """

    def __init__(self, vocab_size, emb_dim, is_sparse=True, padding_idx=0):
        """初始
        """
        super(EmbeddingLayer, self).__init__()
        self.emb_layer = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = emb_dim,
            padding_idx=padding_idx,
            sparse=is_sparse,
            )

    def forward(self, inputs):
        """前向预测
        """
        return self.emb_layer(inputs)


class ConvPoolLayer(nn.Module):
    """卷积池化层
    """
    def __init__(self,
            num_channels,
            num_filters,
            filter_size,
            padding,
            ):
        super(ConvPoolLayer, self).__init__()

        self._conv2d = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=padding,
            )

    def forward(self, inputs):
        """前向预测
        """
        # inputs shape = [batch_size, num_channels, seq_len, emb_dim] [N, C, H, W]
        #logging.info("inputs shape: {}".format(inputs.shape))

        # x shape = [batch_size, num_filters, height_after_conv, width_after_conv=1]
        x = self._conv2d(inputs)
        #logging.info("conv2d shape: {}".format(x.shape))

        # x shape = [batch_size, num_filters, height_after_pool=1, width_after_pool=1]
        x = nn.MaxPool2d(kernel_size=(x.shape[2], 1))(x)
        #logging.info("max pool shape: {}".format(x.shape))

        # x shape = [batch_size, num_filters]
        #x = L.squeeze(x, axes=[2, 3])
        x = torch.reshape(x, shape=(x.shape[0], -1))
        #logging.info("reshape shape: {}".format(x.shape))
        return x


class TextCNNLayer(nn.Module):
    """textcnn层
    """
    def __init__(self,
            emb_dim,
            num_filters=128,
            num_channels=1,
            win_size_list=None,
            output_dim=128,
            ):
        super(TextCNNLayer, self).__init__()

        if win_size_list is None:
            win_size_list = [3]

        def gen_conv_pool(win_size):
            """生成指定窗口的卷积池化层
            """
            return ConvPoolLayer(
                    num_channels,
                    num_filters,
                    (win_size, emb_dim),
                    padding=(1, 0),
                    )

        self.conv_pool_list = nn.ModuleList([gen_conv_pool(win_size) for win_size in win_size_list])

        if output_dim is None:
            output_dim = num_filters * len(win_size_list)

        self.fc = nn.Linear(in_features=num_filters * len(win_size_list), out_features=output_dim)

    def forward(self, input_emb):
        """前向预测
        """
        # emb shape = [batch_size, 1, seq_len, emb_dim]
        emb = torch.unsqueeze(input_emb, dim=1)
        #logging.info("emb shape: {}".format(emb.shape))

        # 列表中各元素的shape = [batch_size, num_filters]
        conv_pool_res_list = [conv_pool(emb) for conv_pool in self.conv_pool_list]
        #for index, conv_pool_res in enumerate(conv_pool_res_list):
        #    logging.info("conv_pool_res #{} shape: {}".format(index, conv_pool_res.shape))

        # 列表中各元素的shape = [batch_size, len(win_size_list)*num_filters]
        conv_pool_res = torch.cat(conv_pool_res_list, dim=-1)

        output = self.fc(conv_pool_res)

        return conv_pool_res


if __name__ == "__main__":
    pass


