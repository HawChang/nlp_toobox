#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   gru_classification.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/07/09 16:29:38
Desc  :   
"""

import torch

from nlp_toolbox.models.base_model import ClassificationModel, model_distributed
from nlp_toolbox.modules.gru import GRUClassifier
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.manual_config import InstanceName


@RegisterSet.models.register
class GRUClassification(ClassificationModel):
    @model_distributed(find_unused_parameters=True, distributed=RegisterSet.IS_DISTRIBUTED)
    def init_model(self, **kwargs):
        model = GRUClassifier(**kwargs)
        return model
