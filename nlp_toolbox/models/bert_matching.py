#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   bert_matching.py
Author:   zhanghao55@baidu.com
Date  :   21/07/15 19:14:51
Desc  :   
"""

import torch

from nlp_toolbox.models.base_model import SimModel, model_distributed
from nlp_toolbox.modules.bert_for_matching import BertForMatching
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.manual_config import InstanceName


@RegisterSet.models.register
class BertMatching(SimModel):
    @model_distributed(find_unused_parameters=True, distributed=RegisterSet.IS_DISTRIBUTED)
    def init_model(self, pretrained_model_dir, **kwargs):
        model = BertForMatching.from_pretrained(
                pretrained_model_dir,
                **kwargs,
                )
        return model
