#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   manual_config.py
Author:   zhanghao55@baidu.com
Date  :   21/06/24 11:03:47
Desc  :   
"""

import sys


class InstanceName(object):
    """InstanceName：常用的变量
    """
    INPUT_IDS = "input_ids"
    INPUT_SEGMENT_IDS = "token_type_ids"
    LABEL_IDS = "labels"

    SECOND_INPUT_IDS = "second_input_ids"
    THIRD_INPUT_IDS = "third_input_ids"

    KEEP_TOKENS = "keep_tokens"
    NUM_CLASS = "num_class"
    VOCAB_SIZE = "vocab_size"

    TOKENIZER = "tokenizer"
    LABEL_ENCODER = "label_encoder"

    MIN_SEQ_LEN = "min_seq_len"
    PAD_IDX = "pad_idx"


class DataLoaderType(object):
    """
    """
    TRAIN_DATALOADER = "train_dataloader"
    TEST_DATALOADER = "test_dataloader"
    EVAL_DATALOADER = "eval_dataloader"
    INFER_DATALOADER = "infer_dataloader"
    TYPE_LIST = [
            TRAIN_DATALOADER,
            TEST_DATALOADER,
            EVAL_DATALOADER,
            INFER_DATALOADER,
            ]


class ImportPackageName(object):
    """ ImportPackageName：要import的文件夹名
    """
    PACKAGE_DIRS = [
            "nlp_toolbox.data.tokenizers",
            "nlp_toolbox.data.data_loaders",
            "nlp_toolbox.data.inputs",
            "nlp_toolbox.models",
            #"nlp_toolbox.data.sampler",
            ]

if __name__ == "__main__":
    pass


