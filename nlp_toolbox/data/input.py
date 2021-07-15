#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   input.py
Author:   zhanghao55@baidu.com
Date  :   21/06/24 18:06:00
Desc  :   
"""


class Field(object):
    """Filed"""
    def __init__(self):
        self.input_by_index = None
        self.input_by_header = None
        self.trans_func = None
        self.trans_config = None

    def build(self, params_dict):
        """
        :param params_dict:
        :return:
        """
        self.name = params_dict["name"]
        self.data_type = params_dict["data_type"]
        self.reader_info = params_dict["reader"]

        self.need_convert = params_dict.get("need_convert", True)
        self.vocab_path = params_dict.get("vocab_path", None)
        self.max_seq_len = params_dict.get("max_seq_len", 512)
        self.truncation_type = params_dict.get("truncation_type", MaxTruncation.KEEP_HEAD)
        self.padding_id = params_dict.get("padding_id", 0)
        self.join_calculation = params_dict.get("join_calculation", True)
        if "num_labels" in params_dict:
            self.num_labels = params_dict["num_labels"]

        # self.label_start_id = params_dict["label_start_id"]
        # self.label_end_id = params_dict["label_end_id"]

        if params_dict.__contains__("embedding"):
            self.embedding_info = params_dict["embedding"]
        if params_dict.__contains__("tokenizer"):
            self.tokenizer_info = params_dict["tokenizer"]
        self.extra_params.update(params_dict.get("extra_params", {}))






