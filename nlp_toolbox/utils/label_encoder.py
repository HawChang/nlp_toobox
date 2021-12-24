#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""
File: label_encoder.py
Author: zhanghao(changhaw@126.com)
Date: 2019/11/08 10:39:02
"""

import codecs

class LabelEncoder(object):
    """类别处理工具
    """
    def __init__(self, label_id_info, other_label_id=0, other_label_name="Other", isFile=True):
        """初始化类别编码类
        [in]  label_id_info: str/dict, 类别及其对应id的信息
              isFile: bool, 说明信息是字典还是文件，若是文件则从其中加载信息
              
        """
        if isFile:
            self.label_id_dict = self.load_class_id_file(label_id_info)
        else:
            if isinstance(label_id_info, dict):
                 self.label_id_dict = label_id_info
            elif isinstance(label_id_info, list) or isinstance(label_id_info, set) or isinstance(label_id_info, tuple):
                 self.label_id_dict = dict()
                 for label in label_id_info:
                     if label in self.label_id_dict:
                         continue
                     self.label_id_dict[label] = len(self.label_id_dict)
            else:
                raise ValueError("unknown label_id_info type: {}".format(type(label_id_info)))
        self.id_label_dict = {v: k for k, v in self.label_id_dict.items()}
        assert other_label_id not in self.id_label_dict, \
            "label_id duplicates the OtherLabelID: {}".format(other_label_id)
        assert other_label_name not in self.label_id_dict, \
            "label_name duplicates the OtherLabelName: {}".format(other_label_name)
        self.other_label_name = other_label_name
        self.other_label_id = other_label_id
        self.label_id_dict[other_label_name] = other_label_id
        self.id_label_dict[other_label_id] = other_label_name
        assert len(self.label_id_dict) == len(self.id_label_dict), "dict is has duplicate key or value."

    def load_class_id_file(self, label_id_path):
        """加载class_id文件
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

    def transform(self, label_name):
        """类别名称转id
        [in]  label_name: str, 类别名称
        [out] label_id: id, 类别名称对应的id
        """
        if label_name not in self.label_id_dict:
            raise ValueError("unknown label name: %s" % label_name)
        return self.label_id_dict[label_name]

    def inverse_transform(self, label_id):
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
