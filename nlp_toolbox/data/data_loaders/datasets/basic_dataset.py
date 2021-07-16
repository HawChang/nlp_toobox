#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   basic_dataset.py
Author:   zhanghao55@baidu.com
Date  :   21/06/25 16:04:58
Desc  :   
"""

import logging
import numpy as np

from torch.utils.data import Dataset


class BasicDataset(Dataset):
    """通用数据集
    """
    def __init__(self, **kwargs):
        """初始化
        """
        ## 一般init函数是加载所有数据
        super(BasicDataset, self).__init__()
        self.data_size = None
        self.data_list_dict = dict()
        for input_name, input_data_list in kwargs.items():
            # dataloader中的数据用numpy保存
            # 相关issue: https://github.com/pytorch/pytorch/issues/13246
            assert self.data_size is None or self.data_size == len(input_data_list), \
                    "inputs' size don't consistant, prev = {}, field {} = {}".format(
                            self.data_size, input_name, len(input_data_list))
            if self.data_size is None:
                self.data_size = len(input_data_list)
                assert self.data_size > 0, "data is empty"
            #if input_name == "second_input_ids":
            #    logging.info("second_input_ids[0] type: {}".format(type(input_data_list[0])))
            input_data_list_np = np.array(input_data_list, dtype=object)
            # np.array时 如果input_data_list中各元素的列表长度一致 则会自动将各元素也变为np.ndarray
            # 这是我们不想要的 我们像input_data_list_np各元素为list
            # 因此给input_data_list第一个列表添加一个元素 转np.array后 再将其去掉 即可得到元素为list的np.array
            if isinstance(input_data_list_np[0], np.ndarray):
                input_data_list[0].append(0)
                input_data_list_np = np.array(input_data_list, dtype=object)
                input_data_list_np[0].pop()
            self.data_list_dict[input_name] = input_data_list_np
            #if input_name == "second_input_ids":
            #    #logging.info("second_input_ids type: {}".format(self.data_list_dict[input_name]))
            #    logging.info("second_input_ids[0] type: {}".format(type(self.data_list_dict[input_name][0])))
            logging.debug("{} shape: {}".format(input_name, self.data_list_dict[input_name].shape))
        logging.debug("data_list shape: ({},{})".format(self.data_size, len(self.data_list_dict.keys())))

    def __getitem__(self, index):
        # 得到单个数据
        output_dict = dict()
        for input_name, input_data_array in self.data_list_dict.items():
            #if input_name == "second_input_ids":
                #logging.info("second_input_ids type: {}".format(input_data_array))
                #logging.info("second_input_ids[0]: {}".format(input_data_array[0]))
                #logging.info("second_input_ids[0] type: {}".format(type(input_data_array[0])))
            output_dict[input_name] = input_data_array[index]
        #logging.info("output dict second_input_ids type: {}".format(type(output_dict["second_input_ids"])))
        return output_dict

    def __len__(self):
        return  0 if self.data_size is None else self.data_size


if __name__ == "__main__":
    pass


