#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   basic_text_input.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/06/25 11:26:38
Desc  :   
"""

from tqdm import tqdm
from collections import defaultdict
from nlp_toolbox.data.inputs.base_input import BaseInput
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.manual_config import InstanceName


@RegisterSet.inputs.register
class BasicTextInput(BaseInput):
    def __init__(self, config):
        super(BasicTextInput, self).__init__(config)
        self.tokenizer = self.config[InstanceName.TOKENIZER]
        self.min_seq_len = self.config.get(InstanceName.MIN_SEQ_LEN, None)
        self.pad_idx = self.config.get(InstanceName.PAD_IDX, 0)

    def encode(self, first_text, second_text=None):
        first_list_length = len(first_text)
        if second_text is None:
            second_text = [None] * first_list_length

        second_list_length = len(second_text)
        assert first_list_length == second_list_length, \
                "first_text length({}) != second_text length({})".format(first_list_length, second_list_length)

        res_dict = defaultdict(list)
        for cur_first_text, cur_second_text in tqdm(zip(first_text, second_text), desc="encode text"):
            cur_token_ids, cur_token_type_ids = self.tokenizer.encode(
                    first_text=cur_first_text,
                    second_text=cur_second_text,
                    )
            if self.min_seq_len is not None and len(cur_token_ids) < self.min_seq_len:
                cur_token_ids += [self.pad_idx] * (self.min_seq_len - len(cur_token_ids))
            if self.min_seq_len is not None and len(cur_token_type_ids) < self.min_seq_len:
                cur_token_type_ids += [self.pad_idx] * (self.min_seq_len - len(cur_token_type_ids))

            res_dict[InstanceName.INPUT_IDS].append(cur_token_ids)
            res_dict[InstanceName.INPUT_SEGMENT_IDS].append(cur_token_type_ids)

        return res_dict
