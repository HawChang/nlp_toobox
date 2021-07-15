#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   textcnn_similarity.py
Author:   zhanghao55@baidu.com
Date  :   21/07/13 19:54:20
Desc  :   
"""

import torch

from nlp_toolbox.models.base_model import SimModel, model_distributed
from nlp_toolbox.modules.textcnn_matching import TextCNNMatching
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.manual_config import InstanceName


@RegisterSet.models.register
class TextCNNSimilarity(SimModel):
    @model_distributed(find_unused_parameters=True, distributed=RegisterSet.IS_DISTRIBUTED)
    def init_model(self, **kwargs):
        model = TextCNNMatching(**kwargs)
        return model
