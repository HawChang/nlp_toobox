#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File  :   test_bert_classifier.py
Author:   zhanghao(changhaw@126.com)
Date  :   21/01/09 14:45:56
Desc  :   
"""

import os
import sys
import codecs
import logging
import torch


_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../../" % _cur_dir)

from nlp_toolbox.utils.logger import init_log
init_log()

from nlp_toolbox.data import data_loader_bundle
from nlp_toolbox.utils import args
from nlp_toolbox.utils import params
from nlp_toolbox.utils import import_modules
from nlp_toolbox.utils.manual_config import InstanceName, DataLoaderType
from nlp_toolbox.utils.register import RegisterSet

RegisterSet.IS_DISTRIBUTED = False
try:
    torch.distributed.init_process_group(backend="nccl")
    LOCAL_RANK = torch.distributed.get_rank()
    RegisterSet.IS_DISTRIBUTED = True

    logging_level = logging.INFO if LOCAL_RANK == 0 else logging.WARNING
    init_log(stream_level=logging_level)
except ValueError as e:
    init_log(stream_level=logging.INFO)

logging.info("is_distributed: {}".format(RegisterSet.IS_DISTRIBUTED))


def dataloader_from_params(params_dict):
    """
    :param params_dict:
    :return: dataloader_bundle
    """
    dataloader_bundle = data_loader_bundle.DataLoaderBundle(params_dict)
    dataloader_bundle.build()

    return dataloader_bundle


def model_from_params(params_dict, dataloader_bundle):
    """
    :param params_dict:
    :return: model
    """
    model_class = RegisterSet.models[params_dict["type"]]

    build_config = params_dict["build_config"]

    for cur_param, cur_value in build_config.items():
        if cur_value != "auto":
            continue
        #if cur_param in ["vocab_size", "num_class", "keep_tokens", "label_encoder", "tokenizer"]:
        build_config[cur_param] = dataloader_bundle.tool_dict[cur_param]

    model = model_class(
            **build_config,
            )

    return model


def run_from_params(params_dict, model, dataloader_bundle):
    """
    """
    phase_list = [
            "train",
            "eval",
            "infer",
            "generate",
            ]

    for cur_phase in phase_list:
        # 每个阶段 都统一开始 
        # 不然train阶段 主进程还在save模型 别的进程就已经到下一个阶段 尝试load模型了
        if RegisterSet.IS_DISTRIBUTED:
            torch.distributed.barrier()

        if cur_phase not in params_dict:
            continue

        cur_config = params_dict[cur_phase]
        if not cur_config.pop("enable", True):
            logging.info("phase {} not enabled, skip".format(cur_phase))
            continue

        logging.info("run at phase: {}".format(cur_phase))

        if cur_phase == "train":
            run_config = {
                    "model_save_path": params_dict["model_save_path"],
                    "best_model_save_path": params_dict["best_model_save_path"],
                    DataLoaderType.TRAIN_DATALOADER: dataloader_bundle[DataLoaderType.TRAIN_DATALOADER].dataloader,
                    DataLoaderType.EVAL_DATALOADER: dataloader_bundle[DataLoaderType.EVAL_DATALOADER].dataloader,
                    }
            run_config.update(params_dict[cur_phase])
        elif cur_phase == "eval":
            model.load_model(params_dict["best_model_save_path"])
            run_config = {DataLoaderType.EVAL_DATALOADER: dataloader_bundle[DataLoaderType.EVAL_DATALOADER].dataloader}
            run_config.update(params_dict[cur_phase])
        elif cur_phase == "infer":
            model.load_model(params_dict["best_model_save_path"])
            run_config = {DataLoaderType.INFER_DATALOADER: dataloader_bundle[DataLoaderType.INFER_DATALOADER].dataloader}
            run_config.update(params_dict[cur_phase])
        elif cur_phase == "generate":
            model.load_model(params_dict["best_model_save_path"])
            run_config = {DataLoaderType.INFER_DATALOADER: dataloader_bundle[DataLoaderType.INFER_DATALOADER].dataloader}
            run_config.update(params_dict[cur_phase])

        logging.info("run config: {}".format(run_config))
        cur_func = getattr(model, cur_phase)
        cur_func(**run_config)


def run_trainer(param_dict):
    """
    :param param_dict:
    :return:
    """
    logging.info("run trainer.... pid = " + str(os.getpid()))
    dataloader_params_dict = param_dict.get("dataloader")
    logging.info("dataloader_params_dict: {}".format(dataloader_params_dict))
    dataloader_bundle = dataloader_from_params(dataloader_params_dict)

    model_params_dict = param_dict.get("model")
    model = model_from_params(model_params_dict, dataloader_bundle)

    run_params_dict = param_dict.get("run_config")
    run_from_params(run_params_dict, model, dataloader_bundle)

    logging.info("end of run train and eval .....")


if __name__ == "__main__":
    # 分布式时 sys.argv会新增一个local_rank参数 解析时需要省略
    parseed_args = args.parse_args(sys.argv[2:]) if RegisterSet.IS_DISTRIBUTED else args.parse_args()

    param_dict = params.from_file(parseed_args.param_path)
    param_dict = params.replace_none(param_dict)
    import_modules.import_modules()
    run_trainer(param_dict)
