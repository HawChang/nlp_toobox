#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   base_input.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/07/01 11:39:41
Desc  :   
"""


class BaseInput(object):
    def __init__(self, config):
        self.config = config

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def upload_attr(self):
        return {}
