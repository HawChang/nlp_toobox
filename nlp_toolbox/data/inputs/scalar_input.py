#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   scalar_input.py
Author:   zhanghao55@baidu.com
Date  :   21/06/25 14:45:22
Desc  :   
"""

from nlp_toolbox.data.inputs.base_input import BaseInput
from nlp_toolbox.utils.manual_config import InstanceName
from nlp_toolbox.utils.register import RegisterSet


class DefaultLabelEncoder(object):
    def __init__(self):
        pass

    def encode(self, label):
        if isinstance(label, str):
            return int(label)


@RegisterSet.inputs.register
class ScalarInput(BaseInput):
    def __init__(self, config, feed_list):
        super(ScalarInput, self).__init__(config, feed_list)
        self.label_encoder = config.get(
                InstanceName.LABEL_ENCODER,
                DefaultLabelEncoder())

    def encode(self, label):
        return {
                InstanceName.LABEL_IDS: [
                    self.label_encoder.encode(cur_label) for cur_label in label
                    ]
                }

    #def upload_attr(self):
    #    return {
    #            InstanceName.LABEL_TOKENIZER: self.label_tokenizer,
    #            InstanceName.NUM_CLASS: self.label_tokenizer.size(),
    #            }
