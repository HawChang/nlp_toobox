#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File  :   register.py
Author:   zhanghao55@baidu.com
Date  :   21/06/24 10:47:36
Desc  :   
"""

import logging


class Register(object):
    """Register"""

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            """decorator"""
            self[key] = value
            return value

        if callable(param):
            logging.info("regist class: {}".format(param.__name__))
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error("module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


class RegisterSet(object):
    """RegisterSet"""
    data_loaders = Register("data_loaders")
    inputs = Register("inputs")
    models = Register("models")
    tokenizers = Register("tokenizers")
    #collate_funcs = Register("collate_funcs")
    #trainer = Register("trainer")
    #embedding = Register("embedding")
    #inference = Register("inference")

    IS_DISTRIBUTED = None
