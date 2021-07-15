#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   bert_classification.py
Author:   zhanghao55@baidu.com
Date  :   21/07/01 14:25:51
Desc  :   
"""

import torch

from nlp_toolbox.models.base_model import ClassificationModel, model_distributed
from nlp_toolbox.modules.bert import BertForClassification
from nlp_toolbox.utils.register import RegisterSet
from nlp_toolbox.utils.manual_config import InstanceName


@RegisterSet.models.register
class BertClassification(ClassificationModel):
    @model_distributed(find_unused_parameters=True, distributed=RegisterSet.IS_DISTRIBUTED)
    def init_model(self, pretrained_model_dir, **kwargs):
        model = BertForClassification.from_pretrained(
                pretrained_model_dir,
                #vocab_size=dataloader_bundle.tool_dict[InstanceName.TOKENIZER].vocab_size,
                #num_class=dataloader_bundle.tool_dict[InstanceName.LABEL_ENCODER].size(),
                #keep_tokens=dataloader_bundle.tool_dict[InstanceName.KEEP_TOKENS],
                **kwargs
                )
        return model

    def init_optimizer(self, model, learning_rate, **kwargs):
        if isinstance(learning_rate, list):
            assert len(learning_rate) == 2, "learning_rate require at most 2 parts, actual {} parts".format(len(learning_rate))
            return torch.optim.Adam([
                {'params': self.get_model().bert.parameters(), 'lr': learning_rate[0]},
                {'params': self.get_model().final_fc.parameters(), 'lr':learning_rate[1]},
                ])
        else:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
