#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   import_modules.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/06/24 10:52:38
Desc  :   
"""

import os

_cur_dir = os.path.dirname(os.path.abspath(__file__))

import importlib
import logging
import traceback

from nlp_toolbox.utils.manual_config import ImportPackageName


def import_modules():
    """import指定包，注册各类
    """
    # 这里只能先找到所有注册的类 之后再import_modules
    # 怀疑原因是：不能在register文件中 import RegisterSet注册的类
    # 不然会由于循环import 而导致注册类无法import RegisterSet
    for package_name in ImportPackageName.PACKAGE_DIRS:
        module_dir = os.path.join(_cur_dir, "../../" + package_name.replace(".", '/'))
        # TODO 只看module_dir根目录下的文件 没有递归 有需要可以修改
        for cur_file_name in os.listdir(module_dir):
            if os.path.isfile(os.path.join(module_dir, cur_file_name)) \
                    and cur_file_name.endswith(".py") \
                    and cur_file_name != "__init__.py":
                cur_module = package_name + "." + cur_file_name[:-3]
                try:
                    importlib.import_module(cur_module)
                except Exception:
                    logging.error("error in import modules")
                    logging.error("traceback.format_exc():\n%s" % traceback.format_exc())


#def import_new_module(package_name, file_name):
#    """import一个新的类
#    :param package_name: 包名
#    :param file_name: 文件名，不需要文件后缀
#    :return:
#    """
#    try:
#        if package_name != "":
#            full_name = package_name + "." + file_name
#        else:
#            full_name = file_name
#        importlib.import_module(full_name)
#
#    except Exception:
#        logging.error("error in import %s" % file_name)
#        logging.error("traceback.format_exc():\n%s" % traceback.format_exc())

