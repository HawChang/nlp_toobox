#!/usr/bin/env python3
# -*- coding:gb18030 -*-
"""
File  :   base_model.py
Author:   zhanghao55@baidu.com
Date  :   20/12/21 10:47:13
Desc  :   
"""

import os
import sys
import codecs
import logging
import math
import numpy as np
import time
import torch

from collections import defaultdict
from sklearn.metrics import classification_report
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from nlp_toolbox.models.adversarial_training import FGM
from nlp_toolbox.utils.manual_config import InstanceName


def model_distributed(local_rank=None, find_unused_parameters=False, distributed=True):
    if distributed:
        if local_rank is None:
            local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        logging.info("set device {} to rank {}".format(device, local_rank))
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_distributed(func):
        def wrapper(self, *args, **kwargs):
            logging.info("model distributed: {}".format(distributed))
            self.device = device
            # 当分布式训练时 local_rank为各进程唯一ID 为0的为主进程
            # 当单机单卡训练时 local_rank为0
            self.local_rank = local_rank
            # 当分布式训练 但该进程不是主进程时 is_master为False，其余情况均为True
            self.is_master = False if local_rank != 0 else True
            model = func(self, *args, **kwargs)
            model.to(self.device)
            if distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=find_unused_parameters)
            # 分布式训练时为True
            self.distributed = distributed
            return model
        return wrapper
    return set_distributed


class BaseModel(object):
    def __init__(self, *args, **kwargs):
        """初始化
        """
        # 初始化模型以及一系列参数
        # 1. self.device: 当前模型所在位置
        # 2. self.distributed: 分布式训练时为True
        # 3. self.local_rank = 0: 当分布式训练时 local_rank为各进程唯一ID 为0的为主进程
        #                         当单机单卡训练时 local_rank为0
        # 4. self.model: 模型
        # 5. self.is_master = True: 当分布式训练 但该进程不是主进程时 is_master为False
        #                           其余情况均为True
        self.model = self.init_model(*args, **kwargs)

    def init_optimizer(self, model, learning_rate, **kwargs):
        """初始化优化器
        """
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    def init_scheduler(self, optimizer, iter_num, scheduler_mode="consine", warm_up=None,
            stepsize=1, gamma=0.9, lr_milestones=None, **kwargs):
        # 判断warm_up配置
        if warm_up is None:
            warm_up_iters = 0
        elif isinstance(warm_up, int):
            warm_up_iters = warm_up
        elif isinstance(warm_up, float):
            warm_up_iters = max(iter_num*warm_up, 1)
        else:
            raise ValueError("expected warm_up in ('None', 'int', 'float'), actual {}".
                    format(type(warm_up)))

        # 学习率变化函数 支持cosine, step, multistep
        def lr_schedule_func(cur_iter):
            res = None
            if cur_iter < warm_up_iters:
                res = (cur_iter + 1) / float(warm_up_iters)
            else:
                total_schedule_num = iter_num - warm_up_iters
                cur_schedule_num = cur_iter - warm_up_iters
                cur_schedule_ratio = cur_schedule_num / total_schedule_num

                if scheduler_mode == "cosine":
                    res = 0.5 * ( math.cos(cur_schedule_ratio * math.pi) + 1)
                elif scheduler_mode == "step":
                    res = gamma**((cur_schedule_ratio // stepsize) * stepsize)
                elif scheduler_mode == "multistep":
                    assert isinstance(lr_milestones, list) or isinstance(lr_milestones, tuple),\
                            "lr_milestones is required as list when scheduler_mode = multistep"
                    res =  gamma**len([m for m in lr_milestones if m <= cur_iter])
            #logging.info("cur iter: {}, scheduler res : {}".format(cur_iter, res))
            return res

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_func)

    def save_model(self, save_path):
        """保存模型
        """
        start_time = time.time()
        torch.save(self.get_model().state_dict(), save_path)
        logging.info("cost time: %.4fs" % (time.time() - start_time))

    def load_model(self, model_path, strict=True):
        """加载模型
        """
        if os.path.exists(model_path):
            logging.info("load model from {}".format(model_path))
            start_time = time.time()
            # 在cpu上加载数据 然后加载到模型
            # 不然在分布式训练时 各卡都会在cuda:0上加载一次数据
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            logging.debug("state_dict_names: {}".format(state_dict.keys()))
            self.get_model().load_state_dict(state_dict, strict=strict)
            torch.cuda.empty_cache()
            logging.info("cost time: %.4fs" % (time.time() - start_time))
        else:
            logging.info("cannot find model file: {}".format(model_path))

    def get_model(self):
        """取得模型
        """
        if self.distributed:
            return self.model.module
        else:
            return self.model

    def train(self, train_dataloader, eval_dataloader,
            model_save_path=None, best_model_save_path=None,
            load_best_model=True, strict=True,
            epochs=5, print_step=50,
            learning_rate=5e-5, scheduler_mode=None,
            adversarial_training=False, swa=False,
            swa_start_epoch=None, swa_lr=None,
            swa_anneal_epoch=5, swa_anneal_strategy="cos",
            **kwargs):
        """ 训练torch模型
        [IN]  train_dataloader: DataLoader, 训练数据
              eval_dataloader: DataLoader, 评估数据
              model_save_path: string, 模型存储路径
              best_model_save_path: string, 最优模型存储路径
              load_best_model: bool, true则加载最优模型
              strict: bool, true则load_state_dict时strict为true
              epochs:  int, 训练轮数
              print_step: int, 每个print_step打印训练情况
              learning_rate: 学习率
              scheduler_mode: string, 学习率模型
              adversarial_training: bool, true则进行参数对抗训练
              swa: bool, true则启动随机权重平均
              swa_start_epoch: int, swa在该epoch开始
              swa_lr: float, swa的目标学习率
              swa_anneal_epoch: int, 从开始的学习率经过swa_start_epoch轮变成目标学习率swa_lr
              swa_anneal_strategy: string, swa转变学习率时的策略
              **kwargs: 其他参数
        [OUT] best_score: float, 训练得到的最优分
        """
        logging.info("train model start at rank {}".format(self.local_rank))
        train_start_time = time.time()

        # 加载最优模型
        if load_best_model:
            self.load_model(best_model_save_path, strict)

        # 参数扰动训练
        if adversarial_training:
            fgm = FGM(self.model)

        # 初始化优化器
        optimizer = self.init_optimizer(self.model, learning_rate, **kwargs)
        # 随机权重平均
        if swa:
            # 为了可以用break快速跳出
            for _ in range(1):
                # 判断epoch数是否够安排swa
                if epochs < 3:
                    logging.warning("epoch num({}) too small to stochastic weight averageing.".
                            format(epochs))
                    swa = False
                    break

                if swa_start_epoch is None:
                    # swa默认从后25%的epoch开始
                    swa_start_epoch = max(int(epochs * 0.75), 2)
                    logging.warning("swa_start_epoch set to {} according to epochs".
                            format(swa_start_epoch))

                if swa_start_epoch > epochs:
                    # epochs >= 2, 因此epochs//2一定大于0
                    new_swa_start_epoch = max(epochs - swa_anneal_epoch, 2)
                    logging.warning("swa_start_epoch({}) > epochs({}), "\
                            "reset swa_start_epoch to {} according to swa_anneal_epoch({})".
                            format(swa_start_epoch, epochs, new_swa_start_epoch, swa_anneal_epoch))
                    swa_start_epoch = new_swa_start_epoch

                if swa_start_epoch < 5:
                    logging.warning("swa is suggested to start when model is stablelized "\
                            "after a series of training(e.g. swa_start_epoch=5). "\
                            "the earlier the swa starts, "\
                            "the higher the probability of model's loss grows unexpectedly")

                if swa_anneal_epoch + swa_start_epoch > epochs:
                    new_swa_anneal_epoch = epochs - swa_start_epoch
                    logging.warning("swa_anneal_epoch({}) + swa_start_epoch({}) > epochs({}),"\
                            "reset swa_anneal_epoch to {}".
                            format(swa_anneal_epoch, swa_start_epoch, epochs, new_swa_anneal_epoch))
                    swa_anneal_epoch = swa_start_epoch

                swa_lr = learning_rate if swa_lr is None else swa_lr

                logging.info("swa_lr = {}".format(swa_lr))
                logging.info("swa_start_epoch = {}".format(swa_start_epoch))
                logging.info("swa_anneal_epoch = {}".format(swa_anneal_epoch))
                logging.info("swa_anneal_strategy = {}".format(swa_anneal_strategy))


                swa_model = AveragedModel(self.model)
                swa_scheduler = SWALR(
                        optimizer,
                        anneal_strategy=swa_anneal_strategy,
                        anneal_epochs=swa_anneal_epoch,
                        swa_lr=swa_lr)

        # 学习率调整
        if scheduler_mode is not None:
            lr_schedule_epoch = swa_start_epoch if swa else epochs
            lr_scheduler = self.init_scheduler(
                    optimizer,
                    lr_schedule_epoch * len(train_dataloader),
                    scheduler_mode=scheduler_mode,
                    **kwargs,
                    )

        cur_train_step = 0
        for cur_epoch in range(epochs):
            # 如果是distributed 要手动给dataloader设置epoch 以让其每个epoch重新打乱数据
            if self.distributed:
                train_dataloader.sampler.set_epoch(cur_epoch)

            # 进入train模式
            # 每epoch都要train 因为eval的时候会变eval
            self.model.train()

            # 主进程的训练展示进度
            if self.is_master:
                pbar = tqdm(total=len(train_dataloader), desc="train progress")

            for cur_train_batch in train_dataloader:
                cur_train_step += 1
                # 清空之前的梯度
                optimizer.zero_grad()

                # 获得本batch_loss 并反传得到梯度
                loss = self.get_loss(**cur_train_batch)
                loss.backward()

                # 对抗训练
                if adversarial_training:
                    # 加入对抗扰动
                    fgm.attack(emb_name="word_embeddings.")
                    # 再计算loss
                    loss_adv = self.get_loss(*cur_train_batch)
                    # 反传 累加梯度
                    loss_adv.backward()
                    # 恢复emb参数
                    fgm.restore()

                # 用获取的梯度更新模型参数
                optimizer.step()
                # 优化器更新后再更新学习率调整器
                # lr_scheduler是每step更新一次
                if (not swa or cur_epoch < swa_start_epoch) and scheduler_mode is not None:
                    lr_scheduler.step()

                logging.debug("optimizer learning_rate: {}".
                        format([x['lr'] for x in optimizer.state_dict()['param_groups']]))

                # 清空之前的梯度
                optimizer.zero_grad()

                loss = loss.cpu().detach().numpy()
                speed = cur_train_step / (time.time() - train_start_time)
                if cur_train_step % print_step == 0:
                    logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                            (cur_epoch, cur_train_step, loss, speed))

                if self.is_master:
                    pbar.set_postfix({
                        "epoch": "{}".format(cur_epoch),
                        "step": "{}".format(cur_train_step),
                        "loss": "{:.5f}".format(loss),
                        "speed": "{:.2f} step/s".format(speed),
                        #"lr ratio": "{:4f}".format(lr_scheduler.get_lr()[1] / float(learning_rate))
                        })
                    pbar.update(1)

            # 因为这里结束后就要下一epoch 因此这里需要判断的是下一个epoch是不是start_epoch
            # 当lr_scheduler结束后 下一个epoch要swa_scheduler了 就应该直接开始step
            # swa_scheduler是每epoch变化一次
            if swa and cur_epoch + 1 >= swa_start_epoch:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()

            if self.is_master:
                # 主进程才有进度展示 关闭
                pbar.close()

                if model_save_path is not None:
                    # 每轮保存模型
                    logging.info("save model at epoch {}".format(cur_epoch))
                    self.save_model(model_save_path + "_epoch{}".format(cur_epoch))

            # 计算验证集准确率
            cur_eval_res = self.eval(eval_dataloader, print_step=print_step, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and is_best and best_model_save_path is not None:
                # 如果是当前最优效果模型 则保存为best模型
                logging.info("cur best score = {}, save model at epoch {} as best model"\
                        .format(self.get_best_score(), cur_epoch))
                self.save_model(best_model_save_path)

        # 所有训练结束后
        # 如果开启了随机权值平均 则最后处理得到swa的模型结果
        if swa:
            update_bn(train_dataloader, swa_model)
            self.model = swa_model

            if self.is_master and model_save_path is not None:
                # 每轮保存模型
                logging.info("save model at swa")
                self.save_model(model_save_path + "_swa")

            # 计算验证集准确率
            cur_eval_res = self.eval(eval_dataloader, print_step=print_step, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and is_best and best_model_save_path is not None:
                # 如果是当前最优效果模型 则保存为best模型
                logging.info("cur best score = {}, save model at swa as best model".format(self.get_best_score()))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

    def single_batch_infer(self, infer_data_dict, is_tensor=True, **kwargs):
        """ 预测单批数据
        [IN]  infer_data_list: list[(input1[, input2, ...])], 待预测数据
              is_tensor: bool, true则输入数据为torch.Tensor, 否则要先转为tensor
        [OUT] infer_res: dict[torch.Tensor], 预测结果
        """
        # infer时不保存反向的梯度
        with torch.no_grad():
            # 控制模型进入eval模式，这将会关闭所有的dropout和norm；
            self.model.eval()

            for k, v in infer_data_dict.items():
                # 如果infer_data_list没有转tensor 则转为torch接收的tensor
                infer_data_dict[k] = v.to(self.device) if is_tensor else torch.tensor(v, device=self.device)

            #logging.info("input ids: {}".format(infer_data_dict["input_ids"]))
            #logging.info("input ids dtype: {}".format(infer_data_dict["input_ids"].dtype))
            #logging.info("pooler_weight: {}".format(infer_data_dict["pooler_weight"]))
            #logging.info("pooler_weight dtype: {}".format(infer_data_dict["pooler_weight"].dtype))
            #if not is_tensor:
            #    infer_data_list = [torch.tensor(x, device=self.device) for x in infer_data_list]
            #else:
            #    infer_data_list = [x.to(self.device) for x in infer_data_list]

            #logging.info("infer_data_list[0] shape: {}".format(infer_data_list[0].shape))
            infer_res = self.model(**infer_data_dict, **kwargs)

            # 按各输出聚合结果
            infer_res = {k: v.detach() for k, v in infer_res.items()}

            #if isinstance(infer_res, tuple):
            #    infer_res = tuple([x.detach() for x in infer_res])
            #else:
            #    infer_res = infer_res.detach()

        return infer_res

    def infer_iter(self, infer_dataloader, print_step=20, fetch_list=None, **kwargs):
        """逐批预测 且每批聚合一次 返回 防止总预测量太大
           WARNING: 慎用！！！该逐批预测合并结果的逻辑与sampler的划分逻辑将导致结果顺序打乱！！！
        """
        # distributed预测时会补齐 因此这里要统计实际预测的数目和数据集补齐前的数目
        actual_infer_num = 0
        origin_infer_num = len(infer_dataloader.dataset)

        cur_infer_step = 0
        cur_infer_time = time.time()
        for cur_infer_tuple in infer_dataloader:
            #if not isinstance(cur_infer_tuple, tuple):
            #    cur_infer_tuple = (cur_infer_tuple,)
            cur_infer_step += 1
            cur_infer_res = self.single_batch_infer(cur_infer_tuple, **kwargs)
            #if not isinstance(cur_logits_tuple, tuple):
            #    cur_logits_tuple = (cur_logits_tuple,)

            if fetch_list is not None:
                cur_infer_res = {x: cur_infer_res[x] for x in fetch_list}
                #if isinstance(gather_output_inds, int):
                #    cur_logits_tuple = (cur_logits_tuple[gather_output_inds],)
                #elif isinstance(gather_output_inds, list) or isinstance(gather_output_inds, tuple):
                #    cur_logits_tuple = [cur_logits_tuple[ind] for ind in gather_output_inds]

            if self.distributed:
                infer_res_dict = dict()
                for k, cur_logits_tensor in cur_infer_res.items():
                    cur_logits_gather = [torch.zeros_like(cur_logits_tensor).to(self.device) \
                            for _ in range(torch.distributed.get_world_size())]
                    # 有gather函数 但对gather操作 nccl只支持all_gather,不支持gather
                    torch.distributed.all_gather(cur_logits_gather, cur_logits_tensor)

                    # 结果拼接
                    cur_logits_gather_tensor = torch.cat(cur_logits_gather, dim=0)
                    logging.debug("cur_logits_gather_tensor shape: {}".format(cur_logits_gather_tensor.shape))

                    # 实际本批数据的数目（去除补齐的数据）
                    cur_actual_infer_num = min(origin_infer_num - actual_infer_num,  len(cur_logits_gather_tensor))

                    # 去除后面补齐的
                    cur_logits_gather_tensor = cur_logits_gather_tensor[:cur_actual_infer_num]
                    logging.debug("cur_logits_gather_tensor strip shape: {}".format(cur_logits_gather_tensor.shape))

                    # 更新当前已预测的数目
                    actual_infer_num += cur_actual_infer_num

                    infer_res_dict[k] = cur_logits_gather_tensor
                    #cur_logits_list.append(cur_logits_gather_tensor)
                    logging.debug("infer_res_dict size: {}".format(len(infer_res_dict)))

                #cur_logits_tuple = tuple(cur_logits_list)
                cur_infer_res = infer_res_dict

            cur_infer_res = {k: v.detach().cpu().numpy() for k, v in cur_infer_res.items()}
            #cur_logits_tuple = tuple([x.detach().cpu().numpy() for x in cur_logits_tuple])

            yield cur_infer_res

            if cur_infer_step % print_step == 0:
                cost_time = time.time() - cur_infer_time
                speed = cur_infer_step / cost_time
                logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                        % (cur_infer_step, cost_time, speed))


    def infer(self, infer_dataloader, print_step=20, fetch_list=None, **kwargs):
        """ 对infer_dataloader进行预测 模型会变为eval状态
        [IN]  infer_dataloader: DataLoader, 待预测数据
              print_step: int, 每个print_step打印训练情况
              gather_output_inds: int or list[int], 指示要获取的输出结果，其他的忽略
        [OUT] pred: tuple(list[float]), 预测结果
        """
        # TODO 各模型的输入输出 需要想个定制化的方法兼容
        # 1. 装饰器
        # 2. get_input、get_output函数重载
        # 3. 参数控制

        infer_res_dict = None

        cur_infer_step = 0
        cur_infer_time = time.time()
        # TODO 采用tqdm展示进度
        for cur_infer_tuple in infer_dataloader:
            cur_infer_step += 1
            ## 输入固定为tuple，预测时*传入
            #if not isinstance(cur_infer_tuple, tuple):
            #    cur_infer_tuple = (cur_infer_tuple,)
            cur_infer_res = self.single_batch_infer(cur_infer_tuple, **kwargs)
            ## 输出固定处理为tuple
            #if not isinstance(cur_logits_tuple, tuple):
            #    cur_logits_tuple = (cur_logits_tuple,)

            # 获取目标输出
            # 因根据序列的预测结果各批次shape不一致 之后不能进行cat
            # 所以这里需要人为指定要获取的输出 剔除各批次shape不一致的结果
            # TODO 想一下上述输出结果shape不一致时不能cat的处理方法 1. 不统一cat?
            if fetch_list is not None:
                cur_infer_res = {x: cur_infer_res[x] for x in fetch_list}
                #if isinstance(gather_output_inds, int):
                #    cur_logits_tuple = (cur_logits_tuple[gather_output_inds],)
                #elif isinstance(gather_output_inds, list) or isinstance(gather_output_inds, tuple):
                #    cur_logits_tuple = [cur_logits_tuple[ind] for ind in gather_output_inds]

            # 若第一次预测 则初始化infer_res_dict
            if infer_res_dict is None:
                infer_res_dict = dict()
                for k in cur_infer_res.keys():
                    infer_res_dict[k] = list()
                #for _ in range(len(cur_logits_tuple)):
                #    infer_res_dict.append(list())

            # 各结果分别添加到各自list中
            for k, v in cur_infer_res.items():
                infer_res_dict[k].append(v.detach())
            #for output_ind, cur_logits in enumerate(cur_logits_tuple):
            #    infer_res_list[output_ind].append(cur_logits.detach())

            # 打印预测信息
            if cur_infer_step % print_step == 0:
                cost_time = time.time() - cur_infer_time
                speed = cur_infer_step / cost_time
                logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                        % (cur_infer_step, cost_time, speed))

        # 拼接预测的各输出tensor
        for k, v in infer_res_dict.items():
            # infer_res_list[index]列表中若tensor shape不一致 则会出错
            infer_res_dict[k] = torch.cat(infer_res_dict[k], dim=0)
            #infer_res_list[index] = torch.cat(infer_res_list[index], dim=0)
            logging.debug("infer_res_dict[{}] shape: {}".format(k, infer_res_dict[k].shape))

        if self.distributed:
            # 如果分布式预测的 需要将结果gather
            infer_res_gather_dict = dict()
            for k, cur_res_tensor in infer_res_dict.items():
                infer_res_gather_dict[k] = self.gather_distributed_tensor(cur_res_tensor, len(infer_dataloader.dataset))
            logging.info("infer_res_gather_dict size: {}".format(len(infer_res_gather_dict)))

            infer_res_dict = infer_res_gather_dict

        # 在这里将数据转为numpy
        # 将各进程数据gather后，一般来说就只需要留主进程做之后的操作了，我们写代码也可以判断当前进程是否主进程，不是则直接退出即可
        # 但其他进程必须在结果数据.detach().cpu().numpy()之后，才能推出，否则程序会卡死
        # 即要主进程将结果转到cpu之后，其余进程才能结束 剩下的操作可由主进程执行
        # 将该操作移到infer函数中 也是怕调用该函数后 程序先结束了其他进程 然后想把结果数据转到cpu而导致程序卡死
        infer_res_dict = {k: v.detach().cpu().numpy() for k, v in infer_res_dict.items()}

        return infer_res_dict

    def gather_distributed_tensor(self, tar_tensor, res_size):
        gather_list = [torch.zeros_like(tar_tensor).to(self.device) for _ in range(torch.distributed.get_world_size())]
        # 有gather函数 但对gather操作 nccl只支持all_gather,不支持gather
        torch.distributed.all_gather(gather_list, tar_tensor)
        # 结果拼接
        gather_tensor = torch.cat(gather_list, dim=0)
        logging.info("gather_tensor shape: {}".format(gather_tensor.shape))
        # 分布式的dataloader会根据batch_size和进程数对数据补齐 使各进程数据能均分
        # 得到结果时需要去除后面补齐的
        gather_tensor = gather_tensor[:res_size]
        logging.info("gather_tensor strip shape: {}".format(gather_tensor.shape))

        return gather_tensor

    def init_model(self, *args, **kwargs):
        """网络构建函数
        """
        raise NotImplementedError

    def get_loss(self, *args, **kwargs):
        """训练时如何得到loss
        """
        raise NotImplementedError

    def eval(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
        """模型评估
        """
        raise NotImplementedError

    def check_if_best(self, cur_eval_res):
        """根据评估结果 判断是否最优
        """
        raise NotImplementedError

    def get_best_score(self):
        """
        """
        raise NotImplementedError


class ClassificationModel(BaseModel):
    def __init__(self, best_acc=None, label_encoder=None, *args, **kwargs):
        """初始化
        """
        super(ClassificationModel, self).__init__(*args, **kwargs)
        self.best_acc = best_acc
        self.label_encoder = label_encoder

    def get_loss(self, **inputs):
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        infer_res = self.model(**inputs)

        loss = infer_res["loss"]

        return loss

    def infer(self, *args, **kwargs):
       infer_res_dict =  super(ClassificationModel, self).infer(*args, **kwargs)

       infer_res_path = kwargs.pop("infer_res_path", None)
       confidence = kwargs.pop("confidence", None)

       if infer_res_path is not None:
           all_pred = np.argmax(infer_res_dict["sent_softmax"], axis=-1)
           all_confidence = np.max(infer_res_dict["sent_softmax"], axis=-1)
           with codecs.open(infer_res_path, "w", "utf-8") as wf:
               for cur_pred, cur_confidence in zip(all_pred, all_confidence):
                   if self.label_encoder is not None:
                       cur_label = self.label_encoder.decode(cur_pred)
                   if confidence is not None and confidence > cur_confidence:
                       cur_confidence = "{}({:.4f})".format(cur_label, cur_confidence)
                       cur_label = "其他"
                   else:
                       cur_confidence = "{:.4f}".format(cur_confidence)
                   wf.write("{}\t{}\n".format(cur_label, cur_confidence))

       return infer_res_dict

    def eval(self, eval_dataloader, print_step=50, **kwargs):
        """
        [IN]  eval_dataloader: DataLoader, 评估数据集
              print_step: int, 每隔print_step展示当前评估信息
        [OUT] acc: float, 分类准确率
        """
        all_pred = list()
        all_label = list()
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        # 验证时不保存反向的梯度
        with torch.no_grad():
            for cur_eval_batch in eval_dataloader:
                cur_eval_step += 1
                cur_label_ids = cur_eval_batch["labels"]
                del cur_eval_batch["labels"]
                cur_logits = self.single_batch_infer(cur_eval_batch)
                cur_pred = cur_logits["sent_softmax"].argmax(dim=-1)
                cur_label = cur_label_ids.detach()
                all_pred.append(cur_pred)
                all_label.append(cur_label)

                if cur_eval_step % print_step == 0:
                    cost_time = time.time() - start_time
                    speed = cur_eval_step / cost_time
                    logging.info('eval step %d, total cost time = %.4fs, speed %.2f step/s' \
                            % (cur_eval_step, cost_time, speed))

        # pred是模型预测的结果 模型是在self.device上的
        all_pred = torch.cat(all_pred, dim=0)
        # label是直接从dataloader拿的数据 还没有放在self.device上
        all_label = torch.cat(all_label, dim=0).to(self.device)

        logging.debug("all pred shape: {}".format(all_pred.shape))
        logging.debug("all label shape: {}".format(all_label.shape))

        if self.distributed:
            all_pred = self.gather_distributed_tensor(all_pred, len(eval_dataloader.dataset))
            all_label = self.gather_distributed_tensor(all_label, len(eval_dataloader.dataset))
            logging.debug("all pred shape: {}".format(all_pred.shape))
            logging.debug("all label shape: {}".format(all_label.shape))

        all_pred = all_pred.cpu().numpy()
        all_label = all_label.cpu().numpy()

        if self.label_encoder is not None:
            all_pred = [self.label_encoder.decode(x) for x in all_pred]
            all_label = [self.label_encoder.decode(x) for x in  all_label]

        logging.debug("eval data size: {}".format(len(all_label)))

        logging.info("\n" + classification_report(all_label, all_pred, digits=4))
        acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
        logging.info("rank {} eval acc : {}".format(self.local_rank, acc))
        return acc

    def check_if_best(self, cur_eval_res):
        """根据评估结果判断是否最优
        [IN]  cur_eval_res: float, 当前评估得分
        [OUT] true则为当前最优得分，否则不是
        """
        if self.best_acc is None or self.best_acc <= cur_eval_res:
            self.best_acc = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        """返回当前最优得分
        """
        return self.best_acc


class SimModel(BaseModel):
    def __init__(self, min_loss=None, *args, **kwargs):
        """初始化
        """
        super(SimModel, self).__init__(*args, **kwargs)
        self.min_loss = min_loss
        self.eval_type = None

    def get_loss(self, **inputs):
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        infer_res = self.model(**inputs)

        loss = infer_res["loss"]

        return loss

    def infer(self, *args, **kwargs):
       infer_res_dict =  super(SimModel, self).infer(*args, **kwargs)

       infer_res_path = kwargs.pop("infer_res_path", None)
       confidence = kwargs.pop("confidence", None)

       if infer_res_path is not None:
           all_sim_score = infer_res_dict["second_sim"]
           all_label = np.where(all_sim_score >= confidence, "1", "0")
           with codecs.open(infer_res_path, "w", "utf-8") as wf:
               for cur_label, cur_sim_score in zip(all_label, all_sim_score):
                   wf.write("{}\t{}\n".format(cur_label, cur_sim_score))

       return infer_res_dict

    def eval(self, eval_dataloader, print_step=50, gather_loss=True, confidence=0.5, **kwargs):
        """
        [IN]  eval_dataloader: DataLoader, 评估数据集
              print_step: int, 每隔print_step展示当前评估信息
        [OUT] acc: float, 分类准确率
        """
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        res_dict = defaultdict(list)

        #loss_list = list()
        #all_pred = list()
        #all_label = list()

        # 主进程的训练展示进度
        if self.is_master:
            pbar = tqdm(total=len(eval_dataloader), desc="eval progress")

        # 验证时不保存反向的梯度
        with torch.no_grad():
            for cur_eval_batch in eval_dataloader:
                cur_eval_step += 1
                # TODO 如果是pointwise 则可以算acc 否则是pairwise 算loss
                # 看cur_eval_batch中有third_input_ids还是labels 有labels则是pointwise 有third_input_ids则是pairwise
                if InstanceName.THIRD_INPUT_IDS in cur_eval_batch:
                    self.eval_type = "pairwise"
                elif InstanceName.LABEL_IDS in cur_eval_batch:
                    self.eval_type = "pointwise"

                if self.eval_type == "pairwise":
                    loss = self.get_loss(**cur_eval_batch)
                    # 保存loss时 先将其detach 不然保存的不只是loss 还有整个计算图
                    res_dict["each_loss"].append(loss.detach().view(-1, 1))
                    #loss_list.append(loss.detach().view(-1, 1))
                elif self.eval_type == "pointwise":
                    cur_label_ids = cur_eval_batch["labels"]
                    cur_label = cur_label_ids.detach()
                    res_dict["label"].append(cur_label.to(self.device))
                    del cur_eval_batch["labels"]

                    cur_infer_res = self.single_batch_infer(cur_eval_batch)
                    cur_pred = cur_infer_res["second_sim"].detach()
                    res_dict["pred"].append(cur_pred)

                if self.is_master:
                    cost_time = time.time() - start_time
                    speed = cur_eval_step / cost_time

                    if cur_eval_step % print_step == 0:
                        logging.info('eval step %d, cost time = %.4fs, speed %.2f step/s' \
                                % (cur_eval_step, cost_time, speed))

                    pbar.set_postfix({
                        "step": "{}".format(cur_eval_step),
                        "cost time": "{:.4f}".format(cost_time),
                        "speed": "{:.2f} step/s".format(speed),
                        })
                    pbar.update(1)

        if self.is_master:
            # 主进程才有进度展示 关闭
            pbar.close()

        for cur_res_name, cur_res_list in res_dict.items():
            cur_res = torch.cat(cur_res_list, dim=0)
            if self.distributed:
                if cur_res_name == "loss":
                    # loss是每批训练数据得到一个平均的loss
                    # 因此其数目应该是各卡的训练批数*卡数
                    # 但因多卡时 会为了给各卡整数批而补齐数据
                    # 因此汇总loss得到平均的loss会和单卡时得到的平均loss不一致
                    # 这是补齐数据导致的
                    # 如果要一致 则需要模型输出各训练数据的loss 即each_loss
                    # 这样按训练数据大小截断时既可以去除补齐数据的loss
                    res_size = len(eval_dataloader) * torch.distributed.get_world_size()
                else:
                    res_size = len(eval_dataloader.dataset)
                cur_res = self.gather_distributed_tensor(cur_res, res_size)
            res_dict[cur_res_name] = cur_res

        if self.is_master:
            if self.eval_type == "pairwise":
                mean_loss = res_dict["each_loss"].mean().cpu().numpy()
                logging.info("rank {} eval loss = {}.".format(self.local_rank, mean_loss))
                return mean_loss

            elif self.eval_type == "pointwise":
                all_pred = res_dict["pred"].cpu().numpy()
                all_pred = np.where(all_pred > confidence, 1, 0)
                all_label = res_dict["label"].cpu().numpy()
                logging.info("\n" + classification_report(all_label, all_pred, digits=4))
                acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
                logging.info("rank {} eval acc : {}".format(self.local_rank, acc))
                return acc


        #if self.eval_type == "pairwise":
        #    total_loss = torch.cat(loss_list, dim=0)
        #    if self.distributed:
        #        total_loss = self.gather_distributed_tensor(total_loss, len(eval_dataloader))
        #    #logging.info("total loss gather : {}".format(total_loss))
        #    total_loss = total_loss.mean().cpu().numpy()
        #    #logging.info("total loss numpy : {}".format(total_loss))
        #    logging.info("rank {} eval loss = {}.".format(self.local_rank, total_loss))

        #    return total_loss

        #elif self.eval_type == "pointwise":
        #    # pred是模型预测的结果 模型是在self.device上的
        #    all_pred = torch.cat(all_pred, dim=0)
        #    # label是直接从dataloader拿的数据 还没有放在self.device上
        #    logging.info("self device: {}".format(self.device))
        #    all_label = torch.cat(all_label, dim=0)
        #    logging.info("all device: {}".format(all_label.device))
        #    all_label = all_label.to(self.device)
        #    logging.info("all device: {}".format(all_label.device))

        #    logging.debug("all pred shape: {}".format(all_pred.shape))
        #    logging.debug("all label shape: {}".format(all_label.shape))

        #    if self.distributed:
        #        all_pred = self.gather_distributed_tensor(all_pred, len(eval_dataloader.dataset))
        #        all_label = self.gather_distributed_tensor(all_label, len(eval_dataloader.dataset))
        #        logging.debug("all pred shape: {}".format(all_pred.shape))
        #        logging.debug("all label shape: {}".format(all_label.shape))

        #    all_pred = all_pred.cpu().numpy()
        #    all_pred = np.where(all_pred > confidence, 1, 0)

        #    all_label = all_label.cpu().numpy()

        #    logging.debug("eval data size: {}".format(len(all_label)))

        #    logging.info("\n" + classification_report(all_label, all_pred, digits=4))
        #    acc = (np.array(all_label) == np.array(all_pred)).astype(np.float32).mean()
        #    logging.info("rank {} eval acc : {}".format(self.local_rank, acc))
        #    return acc

    def check_if_best(self, cur_eval_res):
        """根据评估结果判断是否最优
        [IN]  cur_eval_res: float, 当前评估得分
        [OUT] true则为当前最优得分，否则不是
        """
        if self.eval_type == "pairwise":
            if self.min_loss is None or self.min_loss >= cur_eval_res:
                self.min_loss = cur_eval_res
                return True
            else:
                return False
        elif self.eval_type == "pointwise":
            if self.min_loss is None or self.min_loss <= cur_eval_res:
                self.min_loss = cur_eval_res
                return True
            else:
                return False

    def get_best_score(self):
        """返回当前最优得分
        """
        return self.min_loss


class Seq2seqModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs["tokenizer"]
        super(Seq2seqModel, self).__init__(*args, **kwargs)

    def generate(self, text, out_max_length=40, max_length=512, **kwargs):
        """根据输入文本生成文本
        [IN]  text: str, 文本序列
              out_max_length: int, 输出文本的最长长度
              max_length: int, 输入和输出文本总的最长长度
              **kwargs: beam_search所需参数
        [OUT] generate_text_list: list[str], 生成文本的列表
        """
        # 对 一个 句子生成相应的结果
        ## 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断
        # TODO 应该作为参数 传入beam_search
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length

        # 构造输入数据
        # token_type_id 全为0
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)

        # 生成文本
        generate_list = self.beam_search(
                token_ids,
                token_type_ids,
                self.tokenizer._token_sep_id,
                **kwargs,
                )

        # 对生成的文本解码
        generate_text_list = list()
        for cur_output_ids, cur_score in generate_list:
            cur_text = self.tokenizer.decode(cur_output_ids.detach().cpu().numpy())
            generate_text_list.append((cur_text, cur_score.tolist()))

        return generate_text_list

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=None, min_tokens_to_keep=1):
        """beam_search时按top_k_top_p策略随机抽样输出token
        [IN]  logits: tensor, shape=[beam_size, vocab_size], 当前各beam下各token的输出概率
              top_k: int, 保留概率前top_k的token的概率，其余的概率置零
              top_p: int, 概率由大到小依次累加到刚超过top_p时所覆盖的token保留概率，其余的概率置零
              filter_value: float, 概率置零的token实际在softmax前要改成的值，默认-float('Inf')
        """
        # TODO 直接放参数默认值
        if filter_value is None:
            filter_value = -float('Inf')

        if top_k > 0:
            # 实际的topk要小于实际候选数目
            # 同时保证至少min_tokens_to_keep的抽取概率不为0
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            # 需要置零的token位置为true
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logging.debug("indices_to_remove: {}".format(indices_to_remove))
            # 为true的位置值设为filter_value
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            # 由大到小排序 方便之后累加 保留排序后各位置的原始ind
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            logging.debug("sorted_logits: {}".format(sorted_logits))
            logging.debug("sorted_indices: {}".format(sorted_indices))

            # 累加
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            #cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
            logging.debug("cumulative_probs: {}".format(cumulative_probs))

            # 需要置零的位置为true 这个位置是排序后的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            logging.debug("sorted_indices_to_remove: {}".format(sorted_indices_to_remove))
            if min_tokens_to_keep > 1:
                # 保证至少min_tokens_to_keep的抽取概率不为0
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            logging.debug("sorted_indices_to_remove: {}".format(sorted_indices_to_remove))
            # 右移一位 因为使p刚好大于top_p的那一个也应该保留
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            logging.debug("sorted_indices_to_remove: {}".format(sorted_indices_to_remove))

            # 置零的位置 按sorted_indices转换到排序前的各位置
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logging.debug("indices_to_remove: {}".format(indices_to_remove))
            logits[indices_to_remove] = filter_value

        #candidate_num = (logits != filter_value).sum(dim=-1).item()
        #if candidate_num > 1000:
        #    logging.info("candidate num: {}".format(candidate_num))
        return logits

    def add_penalty(self, score, penalty):
        """为score加惩罚
        [IN]  score: float, 当前得分
              penalty: float, 要加的惩罚 根据值的大小 有不同的惩罚逻辑
        [OUT] score: float, 惩罚后的值
        """
        # penalty为0 表示不变
        if penalty != 0:
            if penalty < 1:
                if score < 0:
                    score /= penalty
                else:
                    score *= penalty
            else:
                score -= penalty
        return score

    # stop_id应该在tokenizer里
    def beam_search(self, token_ids, token_type_ids, stop_id=None,
            beam_size=1, beam_group=1, repeat_penalty=5, diverse_step=5, diverse_penalty=5,
            random_step=1, top_k=0, top_p=1.0, filter_value=None, min_tokens_to_keep=1):
        """beam search操作
        [IN]  token_ids: tensor, 输入的word id
              token_type_ids: tensor, 输入的segment id
              stop_id: int, 作为停止符的token_id，默认是[SEP]的id
              beam_size: int, 指的是各beam_group内的beam_search分支大小
              beam_group: int, beam_group个数
              repeat_penalty: float, 各beam_search分支对输出重复字符的惩罚
              diverse_step: int, diverse周期，每diverse_step个时间步对各group进行diverse惩罚
              diverse_penalty: float, 同时间步内，各group输出其前序group已输出字符一致的惩罚
              random_step: int, 随机抽样周期，每random_step个时间步进行随机抽样
              top_k: int, 随机抽样时，保留输出概率前top_k的token概率
              top_p: int, 随机抽样时，保留输出概率由大到小累积和刚超过top_p时覆盖的token概率
              filter_value: int, 不保留的token的概率改为filter_value，softmax后这些token的概率将变为零
              min_tokens_to_keep: int, 随机抽样时，至少保留min_tokens_to_keep个token的输出概率
        [OUT] generate_list: list[(str, int)], 生成文本及其得分的列表
        """
        if stop_id is None:
            stop_id = self.tokenizer._token_sep_id

        # 一次只输入一个
        # batch_size = 1
        # token_ids shape: [batch_size, seq_length]
        logging.debug("token_ids: {}".format(token_ids))
        logging.debug("token_ids shape: {}".format(token_ids.shape))
        # token_type_ids shape: [batch_size, seq_length]
        logging.debug("token_type_ids : {}".format(token_type_ids))
        logging.debug("token_type_ids  shape: {}".format(token_type_ids.shape))

        # 当前为初始总体beam_size 每次循环时 会随现存beam_group而改变
        total_beam_size = beam_size * beam_group
        logging.debug("total_beam size: {}".format(total_beam_size))
        repeat_word = [list() for i in range(total_beam_size)]

        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=self.device, dtype=torch.long)
        logging.debug("output_ids: {}".format(output_ids))
        logging.debug("output_ids shape: {}".format(output_ids.shape))

        self.model.eval()
        # 记录生成的序列及其得分
        generate_list = list()
        with torch.no_grad():
            # 初始化各得分
            # output_scores shape: [batch_size]
            output_scores = torch.zeros(token_ids.shape[0], device=self.device)
            # 重复生成 直到达到最大长度
            for step in range(self.out_max_length):
                logging.debug("step: {}".format(step))
                #total_beam_size = beam_size * beam_group
                logging.debug("beam size: {}".format(total_beam_size))
                if step == 0:
                    # score shape: [batch_size, seq_length, vocab_size]
                    # TODO 不应该给device参数
                    infer_res = self.model(token_ids, token_type_ids, device=self.device)
                    scores = infer_res["token_output"]
                    logging.debug("scores shape: {}".format(scores.shape))
                    # 第一步只扩充到组 因为后面要找各组topk 如果组里再repeat 就会重复选择一个
                    scores = scores.repeat(beam_group, 1, 1)
                    logging.debug("scores shape: {}".format(scores.shape))

                    # 第一次预测完后才能改变
                    # 重复beam-size次 输入ids
                    # token_ids shape: [total_beam_size, batch_size*seq_length]
                    token_ids = token_ids.view(1, -1).repeat(total_beam_size, 1)
                    logging.debug("token_ids shape: {}".format(token_ids.shape))

                    # token_type_ids shape: [total_beam_size, batch_size*seq_length]
                    token_type_ids = token_type_ids.view(1, -1).repeat(total_beam_size, 1)
                    logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))
                else:
                    # score shape: [total_beam_size, cur_seq_length, vocab_size]
                    # cur_seq_length是逐渐变化的
                    logging.debug("new_input_ids shape: {}".format(new_input_ids.shape))
                    logging.debug("new_token_type_ids shape: {}".format(new_token_type_ids.shape))
                    # TODO 不应该给device参数
                    infer_res = self.model(new_input_ids, new_token_type_ids, device=self.device)
                    scores = infer_res["token_output"]
                    logging.debug("scores shape: {}".format(scores.shape))

                vocab_size = scores.shape[-1]

                # 只取最后一个输出在vocab上的score
                # logit_score shape: step0=[beam_group, vocab_size] other=[total_beam_size, vocab_size]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                # 对每一个已出现过得token降低score
                if repeat_penalty != 0:
                    for i in range(total_beam_size):
                        for token_id in repeat_word[i]:
                            logit_score[i, token_id] = \
                                    self.add_penalty(logit_score[i, token_id], repeat_penalty)

                # logit_score shape: step0=[beam_group, vocab_size] other=[total_beam_size, vocab_size]
                logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                # 取topk的时候我们是组内展平然后再去调用topk函数
                # 同组的各beam结果打平
                # logit_score shape: step0=[beam_group, vocab_size] other=[beam_group, beam_size*vocab_size]
                logit_score = logit_score.view(beam_group, -1)
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                chosen_beam_inds = list()
                chosen_token_inds = list()
                chosen_scores = list()
                # 各group要按先后顺序进行beam_search
                # 因为各group会影响其后续各group
                previous_output_token = set()
                for group_ind, cur_group_score in enumerate(logit_score):
                    logging.debug("group_ind: {}".format(group_ind))
                    # cur_group_score shape: step0=[vocab_size] other=[beam_size*vocab_size]
                    logging.debug("cur_group_score shape: {}".format(cur_group_score.shape))

                    # 根据前序group输出的token添加惩罚
                    if diverse_penalty > 0 and step % diverse_step == diverse_step - 1:
                        # cur_beam_size step0=1, other=beam_size
                        cur_beam_size = cur_group_score.shape[0] // vocab_size
                        for cur_token_id in previous_output_token:
                            for cur_beam_ind in range(cur_beam_size):
                                cur_group_score[cur_token_id + cur_beam_ind * vocab_size] = self.add_penalty(
                                        cur_group_score[cur_token_id + cur_beam_ind * vocab_size], diverse_penalty)

                    # 求当前group各beam输出
                    if random_step > 0 and step % random_step == 0:
                        cur_group_score = self.top_k_top_p_filtering(
                                cur_group_score, top_k, top_p, filter_value, min_tokens_to_keep)

                        cur_group_score = torch.nn.functional.softmax(cur_group_score, dim=-1)
                        logging.debug("cur_group_score: {}".format(cur_group_score))

                        cur_chosen_pos = torch.multinomial(cur_group_score, num_samples=beam_size)

                        # 因cur_group_score是一维数组 所以可以直接用cur_chosen_pos
                        #temp = logits[torch.arange(0, next_tokens.shape[0]).view(-1, 1), cur_chosen_pos]
                        cur_chosen_score = cur_group_score[cur_chosen_pos]
                    else:
                        cur_chosen_score, cur_chosen_pos = torch.topk(cur_group_score, beam_size)

                    logging.debug("cur_chosen_pos: {}".format(cur_chosen_pos))
                    logging.debug("cur_chosen_score: {}".format(cur_chosen_score))
                    logging.debug("cur_chosen_score shape: {}".format(cur_chosen_score.shape))
                    logging.debug("cur_chosen_pos shape: {}".format(cur_chosen_pos.shape))

                    # 记录当前group选中的score
                    chosen_scores.append(cur_chosen_score.view(1, -1))

                    # 记录当前group选中的beam_ind（相对于整体的beam_ind）
                    cur_chosen_beam_inds = (cur_chosen_pos // vocab_size) + group_ind * beam_size
                    logging.debug("cur_chosen_beam_inds: {}".format(cur_chosen_beam_inds))
                    chosen_beam_inds.append(cur_chosen_beam_inds)

                    cur_chosen_token_inds = cur_chosen_pos % vocab_size
                    logging.debug("cur_chosen_token_inds: {}".format(cur_chosen_token_inds))
                    chosen_token_inds.append(cur_chosen_token_inds)

                    # 更新当前group输出的token
                    if diverse_penalty > 0 and step % diverse_step == diverse_step - 1:
                        for token_id in cur_chosen_token_inds:
                            previous_output_token.add(token_id.item())
                    logging.debug("previous_output_token: {}".format(previous_output_token))

                # chosen_beam_inds shape: [total_beam_size]
                chosen_beam_inds = torch.cat(chosen_beam_inds, dim=0)
                logging.debug("chosen_beam_inds: {}".format(chosen_beam_inds))

                # 这个最后要加到output_ids里 所以要变成列向量
                # chosen_token_inds shape: [total_beam_size, 1]
                chosen_token_inds = torch.cat(chosen_token_inds, dim=0).view(-1, 1)
                logging.debug("chosen_token_inds: {}".format(chosen_token_inds))

                # 按组分 这里是各组的top beamsize个分数 之后要看各组的最高分
                # chosen_scores shape: [beam_group, beam_size]
                chosen_scores = torch.cat(chosen_scores, dim=0)
                logging.debug("chosen_scores: {}".format(chosen_scores))

                if repeat_penalty != 0:
                    # 需要有新的repeat_word来更新 不然如果beam中有来自同一种情况的
                    # 第二次遇到时 会把第一次的也加在第二次里
                    new_repeat_word = [list() for i in range(total_beam_size)]
                    logging.debug("repeat_word: {}".format(repeat_word))
                    for index, (beam_ind, word_ind) in enumerate(zip(chosen_beam_inds, chosen_token_inds)):
                        logging.debug("beam_ind: {}".format(beam_ind))
                        logging.debug("word_ind: {}".format(word_ind))
                        new_repeat_word[index] = repeat_word[beam_ind].copy()
                        new_repeat_word[index].append(word_ind.item())
                    repeat_word = new_repeat_word
                    logging.debug("repeat_word: {}".format(repeat_word))

                # 更新得分
                # output_scores shape: [total_beam_size]
                output_scores = chosen_scores.view(-1)
                logging.debug("output_scores: {}".format(output_scores))

                # 更新output_ids
                # 通过chosen_beam_inds选是哪个beam
                # 通过chosen_token_inds选当前beam加哪个token_id
                # output_ids shape: [total_beam_size, cur_seq_length]
                output_ids = torch.cat([output_ids[chosen_beam_inds], chosen_token_inds], dim=1).long()
                logging.debug("output_ids: {}".format(output_ids))

                # new_input_ids shape: [total_beam_size, cur_seq_length]
                # token_ids是固定原输入
                # output_ids是当前beam_search留下的total_beam_size个候选路径
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                logging.debug("new_input_ids shape: {}".format(new_input_ids.shape))

                # new_token_type_ids shape: [total_beam_size, cur_seq_length]
                # token_type_ids后加的type全为1
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                logging.debug("new_token_type_ids shape: {}".format(new_token_type_ids.shape))

                # 记录当前output_ids中有sep_id的情况
                end_counts = (output_ids == stop_id).sum(dim=1)  # 统计出现的end标记
                logging.debug("end_counts: {}".format(end_counts))
                logging.debug("end_counts shape: {}".format(end_counts.shape))
                assert (end_counts < 2).all(), "wrong end_counts: {}".format(end_counts)

                best_beam_inds = chosen_scores.argmax(dim=1)
                logging.debug("best_beam_inds: {}".format(best_beam_inds))
                best_beam_inds += torch.tensor(range(beam_group), device=self.device).view(-1) * beam_size
                logging.debug("best_beam_inds: {}".format(best_beam_inds))

                # 记录各beam是否需要继续
                continue_flag = [True for _ in range(total_beam_size)]

                # 处理已结束的序列
                for group_ind in range(beam_group):
                    for beam_ind in range(beam_size):
                        cur_ind = group_ind * beam_size + beam_ind
                        # 只有结束的beam需要处理
                        if end_counts[cur_ind] > 0:
                            # 如果当前结束的beam同时也是组内score最高的 则该组结束
                            cur_best_ind = best_beam_inds[group_ind]
                            if cur_ind == cur_best_ind:
                                logging.debug("cur_best: {}".format(cur_best_ind))
                                # 加入结果list
                                generate_list.append((
                                    output_ids[cur_best_ind],
                                    output_scores[cur_best_ind].detach().cpu().numpy(),
                                    ))
                                # beam_group这里修改 不会影响range(beam_group)
                                # beam_group数减一
                                beam_group -= 1
                                # 当前组停止
                                for i in range(beam_size * group_ind, beam_size * (group_ind + 1)):
                                    continue_flag[i] = False
                            else:
                                # 否则该beam已结束 但不是组内最高分 将该组最高分copy到当前结束的beam中
                                # 因各beam的token_ids、token_type_ids都一样 所以这里不需要copy
                                new_input_ids[cur_ind] = new_input_ids[cur_best_ind].clone()
                                new_token_type_ids[cur_ind] = new_token_type_ids[cur_best_ind].clone()
                                output_ids[cur_ind] = output_ids[cur_best_ind].clone()
                                output_scores[cur_ind] = output_scores[cur_best_ind].clone()
                                # repeat_word的元素是list 用copy
                                repeat_word[cur_ind] = repeat_word[cur_best_ind].copy()

                logging.debug("continue_flag: {}".format(continue_flag))

                # 去除已完成的序列
                token_ids = token_ids[continue_flag]
                token_type_ids = token_type_ids[continue_flag]
                new_input_ids = new_input_ids[continue_flag]
                new_token_type_ids = new_token_type_ids[continue_flag]
                output_ids = output_ids[continue_flag]
                output_scores = output_scores[continue_flag]
                new_repeat_word = list()
                for index, cur_flag in enumerate(continue_flag):
                    if cur_flag:
                        new_repeat_word.append(repeat_word[index])
                repeat_word = new_repeat_word

                # 下一轮的整体total_beam_size更新
                total_beam_size = beam_size * beam_group
                logging.debug("total beam size: {}".format(total_beam_size))

                # 结束条件 当total_beam_size为0 即beam_group为0时 结束
                if total_beam_size == 0:
                    break

            # generate_list按分排序
            generate_list = sorted(generate_list, key=lambda x:x[1], reverse=True)
            return generate_list


class BertSeq2seqModel(Seq2seqModel):
    def __init__(self, *args, **kwargs):
        """初始化
        """
        super(BertSeq2seqModel, self).__init__(*args, **kwargs)
        self.min_loss = None

    def get_loss(self, **batch):
        for k, v in batch.items():
            batch[k] = v.to(self.device)

        infer_res = self.model(**batch)

        loss = infer_res["loss"]

        return loss

    def train(self, *args, **kwargs):
        # TODO Seq2seqModel应该没有label_encoder
        self.label_encoder = kwargs.pop("label_encoder", None)
        return super(BertSeq2seqModel, self).train(*args, **kwargs)

    # 可能可以放到Seq2seqModel里
    def eval(self, eval_dataloader, print_step=50, gather_loss=True, **kwargs):
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        loss_list = list()
        # 验证时不保存反向的梯度
        with torch.no_grad():
            for batch in eval_dataloader:
                cur_eval_step += 1
                loss = self.get_loss(**batch)
                # 保存loss时 先将其detach 不然保存的不只是loss 还有整个计算图
                loss_list.append(loss.detach().item())
                if cur_eval_step % print_step == 0:
                    cost_time = time.time() - start_time
                    speed = cur_eval_step / cost_time
                    logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                            % (cur_eval_step, cost_time, speed))
        loss_mean = np.mean(loss_list)
        if self.distributed:
            # 当分布式训练时 如果要考虑全部的loss
            # 则如下操作
            if gather_loss:
                loss_tensor = torch.tensor(loss_mean).to(self.device)
                # 这里只打印master进程的loss 所以只需要reduce到rank为0的进程
                # 如果要所有进程loss_tensor同步 用all_reduce
                torch.distributed.reduce(loss_tensor, 0, op=torch.distributed.ReduceOp.SUM)
                if self.is_master:
                    logging.debug("rank {} gather loss total= {}.".format(self.local_rank, loss_tensor))
                    loss_mean = loss_tensor / torch.distributed.get_world_size()
                    logging.info("rank {} gather loss = {}.".format(self.local_rank, loss_mean))
            elif self.is_master:
                # 否则只有master进程打印loss
                logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        else:
            logging.info("eval loss = {}.".format(loss_mean))

        return loss_mean

    def check_if_best(self, cur_eval_res):
        """根据评估结果判断是否最优
        [IN]  cur_eval_res: float, 当前评估得分
        [OUT] true则为当前最优得分，否则不是
        """
        if self.min_loss is None or self.min_loss >= cur_eval_res:
            self.min_loss = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        """返回当前最优得分
        """
        return self.min_loss
