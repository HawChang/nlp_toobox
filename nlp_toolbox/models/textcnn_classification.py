#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   textcnn_classification.py
Author:   zhanghao55@baidu.com
Date  :   21/07/13 11:06:41
Desc  :   
"""

import torch

from nlp_toolbox.models.base_model import ClassificationModel, model_distributed
from nlp_toolbox.modules.textcnn_classifier import TextCNNClassifier
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.manual_config import InstanceName


@RegisterSet.models.register
class TextCNNClassification(ClassificationModel):
    @model_distributed(find_unused_parameters=True, distributed=RegisterSet.IS_DISTRIBUTED)
    def init_model(self, **kwargs):
        model = TextCNNClassifier(**kwargs)
        return model
