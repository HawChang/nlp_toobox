#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   data_loader_bundle.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/06/24 14:15:35
Desc  :   
"""

import logging
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.manual_config import DataLoaderType


class DataLoaderBundle(object):
    """DataLoader"""

    def __init__(self, params_dict):
        """"""
        self.params_dict = params_dict
        self.tool_dict = self._init_tool()

    def _init_tool(self):
        tool_dict = dict()
        for tool_name, tool_config in self.params_dict["tools"].items():
            tool_type = tool_config.pop("type")
            tool_class_name = tool_config.pop("class")
            logging.info("tool_type: {}, tool_class_name: {}".format(tool_type, tool_class_name))
            tool_class = getattr(RegisterSet, tool_type)[tool_class_name]
            tool_res = tool_class.load(**tool_config)
            if isinstance(tool_res, list) or isinstance(tool_res, tuple):
                tool_name_list = tool_name.split(",")
                assert len(tool_name_list) == len(tool_res), \
                        "tools ret size({}) != name size({})".format(
                                len(tool_res), len(tool_name_list))
                for cur_tool_name, cur_tool in zip(tool_name_list, tool_res):
                    tool_dict[cur_tool_name] = cur_tool
            else:
                tool_dict[tool_name] = tool_res
        return tool_dict

    def build(self):
        """构造
        """
        loaders_dict = self.params_dict["loaders"]
        for cur_loader_name in DataLoaderType.TYPE_LIST:
            if cur_loader_name not in loaders_dict:
                self[cur_loader_name] = None
                #setattr(self, cur_loader_name, None)
                continue

            cur_loader_param_dict = loaders_dict[cur_loader_name]

            cur_inputs = cur_loader_param_dict["inputs"]
            cur_config = cur_loader_param_dict["config"]

            cur_loader = RegisterSet.data_loaders[cur_loader_param_dict["type"]](cur_inputs, cur_config, self.tool_dict)
            #cur_loader.build()
            #setattr(self, cur_loader_name, cur_loader)
            self[cur_loader_name] = cur_loader

        #for cur_batch in self[DataLoaderType.TRAIN_DATALOADER].dataloader:
        #    logging.info("cur batch: {}".format(cur_batch))
        #    break

    def to_dict(self):
        """转字典
        """
        return {k: v for k, v in self.items()}

    def items(self):
        """返回k-v对迭代器
        """
        for cur_loader_name in DataLoaderType.TYPE_LIST:
            if cur_loader_name in self:
                yield cur_loader_name, self[cur_loader_name]

    def __setitem__(self, loader_name, loader):
        return setattr(self, loader_name, loader)

    def __getitem__(self, loader_name):
        if hasattr(self, loader_name):
            cur_loader = getattr(self, loader_name)
            if not cur_loader.built:
                cur_loader.build()
            return cur_loader
        else:
            raise KeyError("no loader: {}".format(loader_name))

    def __iter__(self):
        return self

    def __next__(self):
        for cur_loader_name in DataLoaderType.TYPE_LIST:
            if cur_loader_name in self:
                yield self[cur_loader_name]
        raise StopIteration

    def __contains__(self, loader_name):
        return hasattr(cur_loader_name)
