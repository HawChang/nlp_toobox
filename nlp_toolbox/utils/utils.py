#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   utils.py
Author:   zhanghao(changhaw@126.com)
Date  :   20/11/06 15:51:14
Desc  :   
"""

import os
import sys
import logging
import numpy as np
import random
import torch

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

from nlp_toolbox.data.sampler.sequential_distributed_sampler import SequentialDistributedSampler
#from utils.sampler import Sampler
#from utils.data_io import get_attr_values

#_cur_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append("%s/../lib/" % _cur_dir)
#from weighted_sampler import Sampler
#
#sys.path.append("%s/../../../" % _cur_dir)
#from lib.common.data_io import get_attr_values


def check_dir(dir_address):
    """检测目录是否存在
        1. 若不存在则新建
        2. 若存在但不是文件夹，则报错
        3. 若存在且是文件夹则返回
    """
    if not os.path.isdir(dir_address):
        if os.path.exists(dir_address):
            raise ValueError("specified address is not a directory: %s" % dir_address)
        else:
            logging.info("create directory: %s" % dir_address)
            os.makedirs(dir_address)


class ErnieDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data_list):
        """初始化
        """
        ## 一般init函数是加载所有数据
        super(ErnieDataset, self).__init__()
        # dataloader中的数据用numpy保存
        # 相关issue: https://github.com/pytorch/pytorch/issues/13246
        self.data_list = np.array(data_list, dtype=object)
        #LOCAL_RANK = torch.distributed.get_rank() if torch.distributed.is_available() else -1
        #logging.warning("rank #{}, data_list size: {}".format(LOCAL_RANK, len(self.data_list)))

    def __getitem__(self, index):
        # 得到单个数据
        token_ids, token_type_ids, label_id = self.data_list[index]
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "target_id": label_id
        }
        return output

    def __len__(self):
        return len(self.data_list)


class ErnieWeightedDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, data_list):
        """初始化
        """
        ## 一般init函数是加载所有数据
        super(ErnieWeightedDataset, self).__init__()
        # dataloader中的数据用numpy保存
        # 相关issue: https://github.com/pytorch/pytorch/issues/13246
        self.data_list = np.array(data_list, dtype=object)
        #LOCAL_RANK = torch.distributed.get_rank() if torch.distributed.is_available() else -1
        #logging.warning("rank #{}, data_list size: {}".format(LOCAL_RANK, len(self.data_list)))

    def __getitem__(self, index):
        # 得到单个数据
        token_ids, token_type_ids, token_weights, label_id = self.data_list[index]
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "token_weights": token_weights,
            "target_id": label_id
        }
        return output

    def __len__(self):
        return len(self.data_list)


def padding(indice, max_length, pad_idx=0):
    """
    pad 函数
    """
    pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)


def bert_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return {
            "input_ids": token_ids_padded,
            "token_type_ids": token_type_ids_padded,
            "labels": target_ids,
            }


def bert_weighted_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
    token_weights = [data["token_weights"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    token_weights_padded = padding(token_weights, max_length, pad_idx=0).float()

    return {
            "input_ids": token_ids_padded,
            "token_type_ids": token_type_ids_padded,
            "token_weight": token_weights_padded,
            "labels": target_ids,
            }


def bert_infer_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return {
            "input_ids": token_ids_padded,
            "token_type_ids": token_type_ids_padded,
            }


def bert_weighted_infer_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    token_weights = [data["token_weights"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    token_weights_padded = padding(token_weights, max_length, pad_idx=0).float()

    return {
            "input_ids": token_ids_padded,
            "token_type_ids": token_type_ids_padded,
            "token_weight": token_weights_padded,
            }

def common_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    target_ids = [data["target_id"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)

    return {
            "input_ids": token_ids_padded,
            "labels": target_ids,
            }


def common_infer_collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, max_length)

    return {
            "input_ids": token_ids_padded,
            }


def get_dataloader(dataset, collate_fn, batch_size=32, shuffle=True, distributed=False, sampler=None):
    """生成dataloader
    """
    loader_dict = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            }

    if sampler is not None:
        loader_dict["sampler"] = sampler
    elif distributed:
        if shuffle:
            loader_dict["sampler"] = DistributedSampler(dataset)
        else:
            loader_dict["sampler"] = SequentialDistributedSampler(dataset, batch_size=batch_size)
    else:
        loader_dict["shuffle"] = shuffle

    return DataLoader(**loader_dict)


def create_dataset(data_dir, tokenizer, label_encoder,
        max_num=None, each_num="all", other_num="all",
        sim_text_dict=None, refine_label=True,
        test_size=0, random_state=1, encoding="gb18030", example_num=5):
    """生成数据集
    [IN]  data_dir: str，数据文件地址
          tokenizer: object，编码工具
          label_encoder: object，类别映射工具
          max_num: int，数据集样本数最大值
          each_num: int，风险类别每类抽样样本数
          other_num: int，无风险类别每类抽样样本数，label_encoder映射id为0的为无风险，其余有风险
          sim_text_dict: dict，相似样本字典
          refine_label：bool，true则用字典里给出的label，否则相似样本类别与原样本一致
          test_size: float，不为0，则按test_size划分训练集数据集
          random_state: int, 随机划分时的随机种子
          encoding: str, 文件的编码
          example_num: int, 展示数据示例的数目
    """

    fetch_list = ["text", "label"]
    # 加载数据
    fetch_res = get_attr_values(data_dir, fetch_list=fetch_list, encoding=encoding)

    fetch_res = list(zip(*fetch_res))

    if max_num is not None:
        random.shuffle(fetch_res)
        fetch_res = fetch_res[:max_num]
    #logging.info("fetch_res[0]: {}".format(fetch_res[0]))

    has_test = True if test_size > 0 else False

    if has_test:
        train_part, test_part = train_test_split(fetch_res,
                test_size=test_size,
                random_state=random_state,
                shuffle=True)
    else:
        train_part = fetch_res

    # 在train_part的时候 按each_num和other_num抽样
    if each_num != "all" or other_num != "all":
        # each_num或other_num有一个有限制则进行抽样
        if each_num == "all":
            each_num = sys.maxsize
        if other_num == "all":
            other_num = sys.maxsize

        ori_label_count = defaultdict(int)
        sampler_dict = dict()
        for cur_data in train_part:
            cur_label = cur_data[1]
            ori_label_count[cur_label] += 1
            if cur_label not in sampler_dict:
                cur_label_id = label_encoder.transform(cur_label)
                if cur_label_id == 0:
                    sampler_dict[cur_label] = Sampler(other_num)
                else:
                    sampler_dict[cur_label] = Sampler(each_num)
            sampler_dict[cur_label].put(cur_data)
        ori_label_count = sorted(ori_label_count.items(), key=lambda x:x[1], reverse=True)
        ori_label_count = " ".join(["{}({})".format(k, v) for k, v in ori_label_count])
        logging.info("ori label count: {}".format(ori_label_count))

        sample_res = [(label, cur_sampler.get_sample_list()) for label, cur_sampler in sampler_dict.items()]
        sample_label_count = [(label, len(sample_list)) for label, sample_list in sample_res]
        sample_label_count = sorted(sample_label_count, key=lambda x:x[1], reverse=True)
        sample_label_count = " ".join(["{}({})".format(k, v) for k, v in sample_label_count])
        logging.info("sample label count: {}".format(sample_label_count))

        train_part = list()
        for _, sample_list in sample_res:
            train_part.extend(sample_list)

    sim_text_set = set()
    train_text = list()
    train_data_list = list()
    for cur_data in train_part:
        #logging.info("cur_data: {}".format(cur_data))
        cur_text = cur_data[0]
        cur_label = cur_data[1]
        cur_token_ids, cur_token_type_ids = tokenizer.encode(cur_text)
        cur_label_id = label_encoder.transform(cur_label)
        train_data_list.append((cur_token_ids, cur_token_type_ids, cur_label_id))
        train_text.append(cur_text)
        if cur_label_id != 0 and sim_text_dict:
            if cur_text in sim_text_dict:
                #print("\t".join([cur_label, cur_text]))
                sim_text_list = sim_text_dict[cur_text]
                for cur_sim_text, cur_sim_label, _ in sim_text_list:
                    if cur_sim_text in sim_text_set:
                        continue
                    if not refine_label:
                        cur_sim_label = cur_label
                    sim_text_set.add(cur_sim_text)
                    cur_sim_token_ids, cur_sim_token_type_ids = \
                            tokenizer.encode(cur_sim_text)
                    cur_sim_label_id = label_encoder.transform(cur_sim_label)
                    train_data_list.append((
                        cur_sim_token_ids,
                        cur_sim_token_type_ids,
                        cur_sim_label_id,
                        ))
                    train_text.append(cur_sim_text)
                    #print("\t".join([cur_sim_label, cur_sim_text]))
                #print("="*150)
            else:
                logging.warning("not find text in sim_dict: {}".format(cur_text))

    test_data_list = list()
    if has_test:
        for cur_data in test_part:
            cur_text = cur_data[0]
            cur_label = cur_data[1]
            cur_token_ids, cur_token_type_ids = tokenizer.encode(cur_text)
            cur_label_id = label_encoder.transform(cur_label)
            test_data_list.append((cur_token_ids, cur_token_type_ids, cur_label_id))

    logging.info("train num = {}".format(len(train_data_list)))
    logging.info("test num = {}".format(len(test_data_list)))

    train_dataset =  ErnieDataset(train_data_list)
    if has_test:
        test_dataset =  ErnieDataset(test_data_list)

    logging.info(u"数据样例")
    for index, (text, (token_ids, _, label_id)) in enumerate(zip(
            train_text[:example_num],
            train_data_list[:example_num],
            )):
        label_name = label_encoder.inverse_transform(label_id)
        logging.info("example #{}:".format(index))
        logging.info("label: {}".format(label_name))
        logging.info("text: {}".format(text))
        logging.info("token_ids: {}".format(token_ids))

    if has_test:
        return train_dataset, test_dataset
    else:
        return train_dataset


if __name__ == "__main__":
    pass
