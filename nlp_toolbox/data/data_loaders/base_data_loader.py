#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   basic_data_loader.py
Author:   zhanghao55@baidu.com
Date  :   21/06/24 19:39:10
Desc  :   
"""

import sys
import logging

from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.data_io import get_attr_values, get_column_values


class BaseDataLoader(object):
    """基础构造dataloader类
    """
    def __init__(self, inputs, config, tool_dict):
        """
        """
        self.config = config
        self.tool_dict = tool_dict
        self.built = False

        self.inputs = list()
        self._feed_dict_list = list()
        self._fetch_dict_list = list()
        self._feed_name_list = list()
        for cur_input_param_dict in inputs:
            ## self._feed_dict_list[i] = j 表示数据i由j得来
            ## 因为来源j可能重复 而数据i不会重复
            #tar_name_list = list()
            #for src_name, tar_name in cur_input_param_dict["feed_dict"].items():
            #    assert tar_name not in self._feed_dict_list, \
            #            "duplicate tar_name: {}".format(tar_name)
            #    if not self.config["header"]:
            #        src_name = int(src_name)
            #    self._feed_dict_list[tar_name] = src_name
            #    tar_name_list.append(tar_name)

            cur_feed_dict = cur_input_param_dict["feed_dict"]
            if not self.config["header"]:
                cur_feed_dict = {int(k):v for k,v in cur_feed_dict.items()}

            self._feed_name_list.extend(cur_feed_dict.keys())

            cur_input_class = RegisterSet.inputs[cur_input_param_dict["type"]]
            cur_config = cur_input_param_dict["config"]
            need_tool = cur_config.pop("need_tool", list())
            if isinstance(need_tool, str):
                cur_config[need_tool] = self.tool_dict[need_tool]
            elif isinstance(need_tool, list) or isinstance(need_tool, tuple):
                for cur_need_tool in need_tool:
                    cur_config[need_tool] = self.tool_dict[need_tool]

            cur_input = cur_input_class(
                    config=cur_config,
                    #feed_list=tar_name_list,
                    feed_list=None,
                    )
            ## 储存各input的回传属性信息
            #for cur_attr_name, cur_attr_value in cur_input.upload_attr().items():
            #    #logging.info("set {} to {}".format(cur_attr_name, cur_attr_value))
            #    assert not hasattr(self, cur_attr_name), "cur_attr already exists: {}".format(cur_attr_name)
            #    setattr(self, cur_attr_name, cur_attr_value)

            self.inputs.append(cur_input)
            self._fetch_dict_list.append(cur_input_param_dict.get("fetch_dict", None))
            self._feed_dict_list.append(cur_feed_dict)

        # 根据各目标数据的来源确定需要从数据中获取的字段(列)
        self._feed_name_list = list(set(self._feed_name_list))

    def make_input_data(self):
        """构造模型输入数据
        """
        ## 根据各目标数据的来源确定需要从数据中获取的字段(列)
        #src_name_list = list(set(self._feed_dict_list.values()))
        # 数据由表头时 通过字段获取数据 否则通过列获取数据
        fetch_func = get_attr_values if self.config["header"] else get_column_values
        feed_res = fetch_func(
                data_dir=self.config["data_dir"],
                fetch_list=self._feed_name_list,
                encoding=self.config["encoding"],
                )

        # 各来源数据的结果
        feed_name_res_dict = {k:v for k,v in zip(self._feed_name_list, feed_res)}

        ## 从各来源数据确定各目标数据的结果
        #feed_dict = dict()
        #for tar_name in self._feed_dict_list.keys():
        #    feed_dict[tar_name] = src_name_res_dict[self._feed_dict_list[tar_name]]

        # 各类数据处理为模型输入格式
        input_data_dict = dict()
        for cur_input, cur_feed_dict, cur_fetch_dict in zip(self.inputs, self._feed_dict_list, self._fetch_dict_list):
            cur_feed_dict = {cur_feed_dict[cur_feed_name]:feed_name_res_dict[cur_feed_name] \
                    for cur_feed_name in cur_feed_dict.keys()}
            #cur_feed_dict = {cur_feed_name:feed_dict[cur_feed_name] for cur_feed_name in cur_input.feed_list}
            fetch_res_dict = cur_input.encode(**cur_feed_dict)
            #logging.info("fetch_res_dict.keys: {}".format(fetch_res_dict.keys()))
            #logging.info("input_ids example {}".format(fetch_res_dict["input_ids"][:10]))
            #logging.info("fetch second_input_ids type: {}".format(type(fetch_res_dict["second_input_ids"])))

            if cur_fetch_dict is None:
                for cur_input_name, cur_encode_res_list in fetch_res_dict.items():
                    assert cur_input_name not in input_data_dict, "duplicate encode name: {}".format(cur_input_name)
                    input_data_dict[cur_input_name] = cur_encode_res_list
            else:
                for cur_origin_name, cur_modify_name in cur_fetch_dict.items():
                    assert cur_modify_name not in input_data_dict, "duplicate encode name: {}".format(cur_modify_name)
                    input_data_dict[cur_modify_name] = fetch_res_dict[cur_origin_name]
        return input_data_dict

    def build(self):
        """
        """
        self.built = True


if __name__ == "__main__":
    pass


