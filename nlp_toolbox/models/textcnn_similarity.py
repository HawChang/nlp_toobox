#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   textcnn_similarity.py
Author:   zhanghao(changhaw@126.com)
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
    """textcnn 相似匹配函数
    """
    @model_distributed(find_unused_parameters=True, distributed=RegisterSet.IS_DISTRIBUTED)
    def init_model(self, **kwargs):
        """初始化模型
        """
        model = TextCNNMatching(**kwargs)
        return model
