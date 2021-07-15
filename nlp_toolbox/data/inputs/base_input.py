#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   base_input.py
Author:   zhanghao55@baidu.com
Date  :   21/07/01 11:39:41
Desc  :   
"""


class BaseInput(object):
    def __init__(self, config, feed_list):
        self.config = config
        self.feed_list = feed_list

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def upload_attr(self):
        return {}
