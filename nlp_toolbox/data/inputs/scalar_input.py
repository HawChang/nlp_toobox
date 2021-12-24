#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   scalar_input.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/06/25 14:45:22
Desc  :   
"""

from nlp_toolbox.data.inputs.base_input import BaseInput
from nlp_toolbox.utils.manual_config import InstanceName
from nlp_toolbox.utils.register import RegisterSet


@RegisterSet.inputs.register
class ScalarInput(BaseInput):
    def __init__(self, config):
        super(ScalarInput, self).__init__(config)

    def encode(self, label):
        return {InstanceName.LABEL_IDS: \
                [int(cur_label) for cur_label in label]}
