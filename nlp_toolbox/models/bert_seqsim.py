#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   bert_seqsim.py
Author:   zhanghao55@baidu.com
Date  :   21/07/15 19:14:51
Desc  :   
"""

import torch

from nlp_toolbox.models.base_model import BertSeq2seqModel, model_distributed
from nlp_toolbox.modules.bert import BertForSeqSim
from nlp_toolbox.utils.register import RegisterSet


@RegisterSet.models.register
class BertSeqSim(BertSeq2seqModel):
    @model_distributed(find_unused_parameters=True, distributed=RegisterSet.IS_DISTRIBUTED)
    def init_model(self, pretrained_model_dir, **kwargs):
        model = BertForSeqSim.from_pretrained(
                pretrained_model_dir,
                **kwargs,
                )
        return model
