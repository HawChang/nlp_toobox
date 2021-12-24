#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   label_encoder.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/06/25 15:16:00
Desc  :   
"""

import codecs
from nlp_toolbox.utils.register import RegisterSet


@RegisterSet.tokenizers.register
class LabelEncoder(object):
    """label转变
    """
    @classmethod
    def load(cls, vocab_path):
        """加载
        """
        label_id_dict = cls.load_label_id(vocab_path)
        label_tokenizer = cls(label_id_dict)
        return label_tokenizer, label_tokenizer.size()

    @classmethod
    def load_label_id(cls, label_id_path):
        """加载label_id文件
        [in]  label_id_path: str, 类别及其对应id信息文件
        [out] label_id_dict: dict, 类别->id字典
        """
        id_set = set()
        label_id_dict = dict()
        with codecs.open(label_id_path, "r", "gb18030") as rf:
            for line in rf:
                parts = line.strip("\n").split("\t")
                label_id = int(parts[0])
                label_name = parts[1]
                assert label_name not in label_id_dict, "duplicate label_name(%s)" % label_name
                assert label_id not in id_set, "duplicate label_id(%d)" % label_id
                label_id_dict[label_name] = label_id
        return label_id_dict

    def __init__(self, label_id_dict):
        """初始化类别编码类
        [in]  label_id_dict: dict, 类别及其对应id的信息
        """
        self.label_id_dict = label_id_dict
        self.id_label_dict = {v: k for k, v in self.label_id_dict.items()}
        assert len(self.label_id_dict) == len(self.id_label_dict), "dict is has duplicate key or value."

    def encode(self, label_name):
        """类别名称转id
        [in]  label_name: str, 类别名称
        [out] label_id: id, 类别名称对应的id
        """
        if label_name not in self.label_id_dict:
            raise ValueError("unknown label name: %s" % label_name)
        return self.label_id_dict[label_name]

    def decode(self, label_id):
        """类别名称转id
        [in]  label_id: id, 类别名称对应的id
        [out] label_name: str, 类别名称
        """
        if label_id not in self.id_label_dict:
            raise ValueError("unknown label id: %s" % label_id)
        return self.id_label_dict[label_id]

    def size(self):
        """返回类别数
        [out] label_num: int, 类别树目
        """
        return len(self.label_id_dict)
