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
            # ���ֲ�ʽѵ��ʱ local_rankΪ������ΨһID Ϊ0��Ϊ������
            # ����������ѵ��ʱ local_rankΪ0
            self.local_rank = local_rank
            # ���ֲ�ʽѵ�� ���ý��̲���������ʱ is_masterΪFalse�����������ΪTrue
            self.is_master = False if local_rank != 0 else True
            model = func(self, *args, **kwargs)
            model.to(self.device)
            if distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=find_unused_parameters)
            # �ֲ�ʽѵ��ʱΪTrue
            self.distributed = distributed
            return model
        return wrapper
    return set_distributed


class BaseModel(object):
    def __init__(self, *args, **kwargs):
        """��ʼ��
        """
        # ��ʼ��ģ���Լ�һϵ�в���
        # 1. self.device: ��ǰģ������λ��
        # 2. self.distributed: �ֲ�ʽѵ��ʱΪTrue
        # 3. self.local_rank = 0: ���ֲ�ʽѵ��ʱ local_rankΪ������ΨһID Ϊ0��Ϊ������
        #                         ����������ѵ��ʱ local_rankΪ0
        # 4. self.model: ģ��
        # 5. self.is_master = True: ���ֲ�ʽѵ�� ���ý��̲���������ʱ is_masterΪFalse
        #                           ���������ΪTrue
        self.model = self.init_model(*args, **kwargs)

    def init_optimizer(self, model, learning_rate, **kwargs):
        """��ʼ���Ż���
        """
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    def init_scheduler(self, optimizer, iter_num, scheduler_mode="consine", warm_up=None,
            stepsize=1, gamma=0.9, lr_milestones=None, **kwargs):
        # �ж�warm_up����
        if warm_up is None:
            warm_up_iters = 0
        elif isinstance(warm_up, int):
            warm_up_iters = warm_up
        elif isinstance(warm_up, float):
            warm_up_iters = max(iter_num*warm_up, 1)
        else:
            raise ValueError("expected warm_up in ('None', 'int', 'float'), actual {}".
                    format(type(warm_up)))

        # ѧϰ�ʱ仯���� ֧��cosine, step, multistep
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
        """����ģ��
        """
        start_time = time.time()
        torch.save(self.get_model().state_dict(), save_path)
        logging.info("cost time: %.4fs" % (time.time() - start_time))

    def load_model(self, model_path, strict=True):
        """����ģ��
        """
        if os.path.exists(model_path):
            logging.info("load model from {}".format(model_path))
            start_time = time.time()
            # ��cpu�ϼ������� Ȼ����ص�ģ��
            # ��Ȼ�ڷֲ�ʽѵ��ʱ ����������cuda:0�ϼ���һ������
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            logging.debug("state_dict_names: {}".format(state_dict.keys()))
            self.get_model().load_state_dict(state_dict, strict=strict)
            torch.cuda.empty_cache()
            logging.info("cost time: %.4fs" % (time.time() - start_time))
        else:
            logging.info("cannot find model file: {}".format(model_path))

    def get_model(self):
        """ȡ��ģ��
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
        """ ѵ��torchģ��
        [IN]  train_dataloader: DataLoader, ѵ������
              eval_dataloader: DataLoader, ��������
              model_save_path: string, ģ�ʹ洢·��
              best_model_save_path: string, ����ģ�ʹ洢·��
              load_best_model: bool, true���������ģ��
              strict: bool, true��load_state_dictʱstrictΪtrue
              epochs:  int, ѵ������
              print_step: int, ÿ��print_step��ӡѵ�����
              learning_rate: ѧϰ��
              scheduler_mode: string, ѧϰ��ģ��
              adversarial_training: bool, true����в����Կ�ѵ��
              swa: bool, true���������Ȩ��ƽ��
              swa_start_epoch: int, swa�ڸ�epoch��ʼ
              swa_lr: float, swa��Ŀ��ѧϰ��
              swa_anneal_epoch: int, �ӿ�ʼ��ѧϰ�ʾ���swa_start_epoch�ֱ��Ŀ��ѧϰ��swa_lr
              swa_anneal_strategy: string, swaת��ѧϰ��ʱ�Ĳ���
              **kwargs: ��������
        [OUT] best_score: float, ѵ���õ������ŷ�
        """
        logging.info("train model start at rank {}".format(self.local_rank))
        train_start_time = time.time()

        # ��������ģ��
        if load_best_model:
            self.load_model(best_model_save_path, strict)

        # �����Ŷ�ѵ��
        if adversarial_training:
            fgm = FGM(self.model)

        # ��ʼ���Ż���
        optimizer = self.init_optimizer(self.model, learning_rate, **kwargs)
        # ���Ȩ��ƽ��
        if swa:
            # Ϊ�˿�����break��������
            for _ in range(1):
                # �ж�epoch���Ƿ񹻰���swa
                if epochs < 3:
                    logging.warning("epoch num({}) too small to stochastic weight averageing.".
                            format(epochs))
                    swa = False
                    break

                if swa_start_epoch is None:
                    # swaĬ�ϴӺ�25%��epoch��ʼ
                    swa_start_epoch = max(int(epochs * 0.75), 2)
                    logging.warning("swa_start_epoch set to {} according to epochs".
                            format(swa_start_epoch))

                if swa_start_epoch > epochs:
                    # epochs >= 2, ���epochs//2һ������0
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

        # ѧϰ�ʵ���
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
            # �����distributed Ҫ�ֶ���dataloader����epoch ������ÿ��epoch���´�������
            if self.distributed:
                train_dataloader.sampler.set_epoch(cur_epoch)

            # ����trainģʽ
            # ÿepoch��Ҫtrain ��Ϊeval��ʱ����eval
            self.model.train()

            # �����̵�ѵ��չʾ����
            if self.is_master:
                pbar = tqdm(total=len(train_dataloader), desc="train progress")

            for cur_train_batch in train_dataloader:
                cur_train_step += 1
                # ���֮ǰ���ݶ�
                optimizer.zero_grad()

                # ��ñ�batch_loss �������õ��ݶ�
                loss = self.get_loss(**cur_train_batch)
                loss.backward()

                # �Կ�ѵ��
                if adversarial_training:
                    # ����Կ��Ŷ�
                    fgm.attack(emb_name="word_embeddings.")
                    # �ټ���loss
                    loss_adv = self.get_loss(*cur_train_batch)
                    # ���� �ۼ��ݶ�
                    loss_adv.backward()
                    # �ָ�emb����
                    fgm.restore()

                # �û�ȡ���ݶȸ���ģ�Ͳ���
                optimizer.step()
                # �Ż������º��ٸ���ѧϰ�ʵ�����
                # lr_scheduler��ÿstep����һ��
                if (not swa or cur_epoch < swa_start_epoch) and scheduler_mode is not None:
                    lr_scheduler.step()

                logging.debug("optimizer learning_rate: {}".
                        format([x['lr'] for x in optimizer.state_dict()['param_groups']]))

                # ���֮ǰ���ݶ�
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

            # ��Ϊ����������Ҫ��һepoch ���������Ҫ�жϵ�����һ��epoch�ǲ���start_epoch
            # ��lr_scheduler������ ��һ��epochҪswa_scheduler�� ��Ӧ��ֱ�ӿ�ʼstep
            # swa_scheduler��ÿepoch�仯һ��
            if swa and cur_epoch + 1 >= swa_start_epoch:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()

            if self.is_master:
                # �����̲��н���չʾ �ر�
                pbar.close()

                if model_save_path is not None:
                    # ÿ�ֱ���ģ��
                    logging.info("save model at epoch {}".format(cur_epoch))
                    self.save_model(model_save_path + "_epoch{}".format(cur_epoch))

            # ������֤��׼ȷ��
            cur_eval_res = self.eval(eval_dataloader, print_step=print_step, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and is_best and best_model_save_path is not None:
                # ����ǵ�ǰ����Ч��ģ�� �򱣴�Ϊbestģ��
                logging.info("cur best score = {}, save model at epoch {} as best model"\
                        .format(self.get_best_score(), cur_epoch))
                self.save_model(best_model_save_path)

        # ����ѵ��������
        # ������������Ȩֵƽ�� �������õ�swa��ģ�ͽ��
        if swa:
            update_bn(train_dataloader, swa_model)
            self.model = swa_model

            if self.is_master and model_save_path is not None:
                # ÿ�ֱ���ģ��
                logging.info("save model at swa")
                self.save_model(model_save_path + "_swa")

            # ������֤��׼ȷ��
            cur_eval_res = self.eval(eval_dataloader, print_step=print_step, **kwargs)
            is_best = self.check_if_best(cur_eval_res)
            if self.is_master and is_best and best_model_save_path is not None:
                # ����ǵ�ǰ����Ч��ģ�� �򱣴�Ϊbestģ��
                logging.info("cur best score = {}, save model at swa as best model".format(self.get_best_score()))
                self.save_model(best_model_save_path)
        logging.info("train model cost time %.4fs" % (time.time() - train_start_time))
        return self.get_best_score()

    def single_batch_infer(self, infer_data_dict, is_tensor=True, **kwargs):
        """ Ԥ�ⵥ������
        [IN]  infer_data_list: list[(input1[, input2, ...])], ��Ԥ������
              is_tensor: bool, true����������Ϊtorch.Tensor, ����Ҫ��תΪtensor
        [OUT] infer_res: dict[torch.Tensor], Ԥ����
        """
        # inferʱ�����淴����ݶ�
        with torch.no_grad():
            # ����ģ�ͽ���evalģʽ���⽫��ر����е�dropout��norm��
            self.model.eval()

            for k, v in infer_data_dict.items():
                # ���infer_data_listû��תtensor ��תΪtorch���յ�tensor
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

            # ��������ۺϽ��
            infer_res = {k: v.detach() for k, v in infer_res.items()}

            #if isinstance(infer_res, tuple):
            #    infer_res = tuple([x.detach() for x in infer_res])
            #else:
            #    infer_res = infer_res.detach()

        return infer_res

    def infer_iter(self, infer_dataloader, print_step=20, fetch_list=None, **kwargs):
        """����Ԥ�� ��ÿ���ۺ�һ�� ���� ��ֹ��Ԥ����̫��
           WARNING: ���ã�����������Ԥ��ϲ�������߼���sampler�Ļ����߼������½��˳����ң�����
        """
        # distributedԤ��ʱ�Ჹ�� �������Ҫͳ��ʵ��Ԥ�����Ŀ�����ݼ�����ǰ����Ŀ
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
                    # ��gather���� ����gather���� ncclֻ֧��all_gather,��֧��gather
                    torch.distributed.all_gather(cur_logits_gather, cur_logits_tensor)

                    # ���ƴ��
                    cur_logits_gather_tensor = torch.cat(cur_logits_gather, dim=0)
                    logging.debug("cur_logits_gather_tensor shape: {}".format(cur_logits_gather_tensor.shape))

                    # ʵ�ʱ������ݵ���Ŀ��ȥ����������ݣ�
                    cur_actual_infer_num = min(origin_infer_num - actual_infer_num,  len(cur_logits_gather_tensor))

                    # ȥ�����油���
                    cur_logits_gather_tensor = cur_logits_gather_tensor[:cur_actual_infer_num]
                    logging.debug("cur_logits_gather_tensor strip shape: {}".format(cur_logits_gather_tensor.shape))

                    # ���µ�ǰ��Ԥ�����Ŀ
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
        """ ��infer_dataloader����Ԥ�� ģ�ͻ��Ϊeval״̬
        [IN]  infer_dataloader: DataLoader, ��Ԥ������
              print_step: int, ÿ��print_step��ӡѵ�����
              gather_output_inds: int or list[int], ָʾҪ��ȡ���������������ĺ���
        [OUT] pred: tuple(list[float]), Ԥ����
        """
        # TODO ��ģ�͵�������� ��Ҫ������ƻ��ķ�������
        # 1. װ����
        # 2. get_input��get_output��������
        # 3. ��������

        infer_res_dict = None

        cur_infer_step = 0
        cur_infer_time = time.time()
        # TODO ����tqdmչʾ����
        for cur_infer_tuple in infer_dataloader:
            cur_infer_step += 1
            ## ����̶�Ϊtuple��Ԥ��ʱ*����
            #if not isinstance(cur_infer_tuple, tuple):
            #    cur_infer_tuple = (cur_infer_tuple,)
            cur_infer_res = self.single_batch_infer(cur_infer_tuple, **kwargs)
            ## ����̶�����Ϊtuple
            #if not isinstance(cur_logits_tuple, tuple):
            #    cur_logits_tuple = (cur_logits_tuple,)

            # ��ȡĿ�����
            # ��������е�Ԥ����������shape��һ�� ֮���ܽ���cat
            # ����������Ҫ��Ϊָ��Ҫ��ȡ����� �޳�������shape��һ�µĽ��
            # TODO ��һ������������shape��һ��ʱ����cat�Ĵ����� 1. ��ͳһcat?
            if fetch_list is not None:
                cur_infer_res = {x: cur_infer_res[x] for x in fetch_list}
                #if isinstance(gather_output_inds, int):
                #    cur_logits_tuple = (cur_logits_tuple[gather_output_inds],)
                #elif isinstance(gather_output_inds, list) or isinstance(gather_output_inds, tuple):
                #    cur_logits_tuple = [cur_logits_tuple[ind] for ind in gather_output_inds]

            # ����һ��Ԥ�� ���ʼ��infer_res_dict
            if infer_res_dict is None:
                infer_res_dict = dict()
                for k in cur_infer_res.keys():
                    infer_res_dict[k] = list()
                #for _ in range(len(cur_logits_tuple)):
                #    infer_res_dict.append(list())

            # ������ֱ���ӵ�����list��
            for k, v in cur_infer_res.items():
                infer_res_dict[k].append(v.detach())
            #for output_ind, cur_logits in enumerate(cur_logits_tuple):
            #    infer_res_list[output_ind].append(cur_logits.detach())

            # ��ӡԤ����Ϣ
            if cur_infer_step % print_step == 0:
                cost_time = time.time() - cur_infer_time
                speed = cur_infer_step / cost_time
                logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                        % (cur_infer_step, cost_time, speed))

        # ƴ��Ԥ��ĸ����tensor
        for k, v in infer_res_dict.items():
            # infer_res_list[index]�б�����tensor shape��һ�� ������
            infer_res_dict[k] = torch.cat(infer_res_dict[k], dim=0)
            #infer_res_list[index] = torch.cat(infer_res_list[index], dim=0)
            logging.debug("infer_res_dict[{}] shape: {}".format(k, infer_res_dict[k].shape))

        if self.distributed:
            # ����ֲ�ʽԤ��� ��Ҫ�����gather
            infer_res_gather_dict = dict()
            for k, cur_res_tensor in infer_res_dict.items():
                infer_res_gather_dict[k] = self.gather_distributed_tensor(cur_res_tensor, len(infer_dataloader.dataset))
            logging.info("infer_res_gather_dict size: {}".format(len(infer_res_gather_dict)))

            infer_res_dict = infer_res_gather_dict

        # �����ｫ����תΪnumpy
        # ������������gather��һ����˵��ֻ��Ҫ����������֮��Ĳ����ˣ�����д����Ҳ�����жϵ�ǰ�����Ƿ������̣�������ֱ���˳�����
        # ���������̱����ڽ������.detach().cpu().numpy()֮�󣬲����Ƴ����������Ῠ��
        # ��Ҫ�����̽����ת��cpu֮��������̲��ܽ��� ʣ�µĲ�������������ִ��
        # ���ò����Ƶ�infer������ Ҳ���µ��øú����� �����Ƚ������������� Ȼ����ѽ������ת��cpu�����³�����
        infer_res_dict = {k: v.detach().cpu().numpy() for k, v in infer_res_dict.items()}

        return infer_res_dict

    def gather_distributed_tensor(self, tar_tensor, res_size):
        gather_list = [torch.zeros_like(tar_tensor).to(self.device) for _ in range(torch.distributed.get_world_size())]
        # ��gather���� ����gather���� ncclֻ֧��all_gather,��֧��gather
        torch.distributed.all_gather(gather_list, tar_tensor)
        # ���ƴ��
        gather_tensor = torch.cat(gather_list, dim=0)
        logging.info("gather_tensor shape: {}".format(gather_tensor.shape))
        # �ֲ�ʽ��dataloader�����batch_size�ͽ����������ݲ��� ʹ�����������ܾ���
        # �õ����ʱ��Ҫȥ�����油���
        gather_tensor = gather_tensor[:res_size]
        logging.info("gather_tensor strip shape: {}".format(gather_tensor.shape))

        return gather_tensor

    def init_model(self, *args, **kwargs):
        """���繹������
        """
        raise NotImplementedError

    def get_loss(self, *args, **kwargs):
        """ѵ��ʱ��εõ�loss
        """
        raise NotImplementedError

    def eval(self, eval_dataloader, print_step=50, gather_loss=False, **kwargs):
        """ģ������
        """
        raise NotImplementedError

    def check_if_best(self, cur_eval_res):
        """����������� �ж��Ƿ�����
        """
        raise NotImplementedError

    def get_best_score(self):
        """
        """
        raise NotImplementedError


class ClassificationModel(BaseModel):
    def __init__(self, best_acc=None, label_encoder=None, *args, **kwargs):
        """��ʼ��
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
                       cur_label = "����"
                   else:
                       cur_confidence = "{:.4f}".format(cur_confidence)
                   wf.write("{}\t{}\n".format(cur_label, cur_confidence))

       return infer_res_dict

    def eval(self, eval_dataloader, print_step=50, **kwargs):
        """
        [IN]  eval_dataloader: DataLoader, �������ݼ�
              print_step: int, ÿ��print_stepչʾ��ǰ������Ϣ
        [OUT] acc: float, ����׼ȷ��
        """
        all_pred = list()
        all_label = list()
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        # ��֤ʱ�����淴����ݶ�
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

        # pred��ģ��Ԥ��Ľ�� ģ������self.device�ϵ�
        all_pred = torch.cat(all_pred, dim=0)
        # label��ֱ�Ӵ�dataloader�õ����� ��û�з���self.device��
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
        """������������ж��Ƿ�����
        [IN]  cur_eval_res: float, ��ǰ�����÷�
        [OUT] true��Ϊ��ǰ���ŵ÷֣�������
        """
        if self.best_acc is None or self.best_acc <= cur_eval_res:
            self.best_acc = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        """���ص�ǰ���ŵ÷�
        """
        return self.best_acc


class SimModel(BaseModel):
    def __init__(self, min_loss=None, *args, **kwargs):
        """��ʼ��
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
        [IN]  eval_dataloader: DataLoader, �������ݼ�
              print_step: int, ÿ��print_stepչʾ��ǰ������Ϣ
        [OUT] acc: float, ����׼ȷ��
        """
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        res_dict = defaultdict(list)

        #loss_list = list()
        #all_pred = list()
        #all_label = list()

        # �����̵�ѵ��չʾ����
        if self.is_master:
            pbar = tqdm(total=len(eval_dataloader), desc="eval progress")

        # ��֤ʱ�����淴����ݶ�
        with torch.no_grad():
            for cur_eval_batch in eval_dataloader:
                cur_eval_step += 1
                # TODO �����pointwise �������acc ������pairwise ��loss
                # ��cur_eval_batch����third_input_ids����labels ��labels����pointwise ��third_input_ids����pairwise
                if InstanceName.THIRD_INPUT_IDS in cur_eval_batch:
                    self.eval_type = "pairwise"
                elif InstanceName.LABEL_IDS in cur_eval_batch:
                    self.eval_type = "pointwise"

                if self.eval_type == "pairwise":
                    loss = self.get_loss(**cur_eval_batch)
                    # ����lossʱ �Ƚ���detach ��Ȼ����Ĳ�ֻ��loss ������������ͼ
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
            # �����̲��н���չʾ �ر�
            pbar.close()

        for cur_res_name, cur_res_list in res_dict.items():
            cur_res = torch.cat(cur_res_list, dim=0)
            if self.distributed:
                if cur_res_name == "loss":
                    # loss��ÿ��ѵ�����ݵõ�һ��ƽ����loss
                    # �������ĿӦ���Ǹ�����ѵ������*����
                    # ����࿨ʱ ��Ϊ�˸���������������������
                    # ��˻���loss�õ�ƽ����loss��͵���ʱ�õ���ƽ��loss��һ��
                    # ���ǲ������ݵ��µ�
                    # ���Ҫһ�� ����Ҫģ�������ѵ�����ݵ�loss ��each_loss
                    # ������ѵ�����ݴ�С�ض�ʱ�ȿ���ȥ���������ݵ�loss
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
        #    # pred��ģ��Ԥ��Ľ�� ģ������self.device�ϵ�
        #    all_pred = torch.cat(all_pred, dim=0)
        #    # label��ֱ�Ӵ�dataloader�õ����� ��û�з���self.device��
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
        """������������ж��Ƿ�����
        [IN]  cur_eval_res: float, ��ǰ�����÷�
        [OUT] true��Ϊ��ǰ���ŵ÷֣�������
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
        """���ص�ǰ���ŵ÷�
        """
        return self.min_loss


class Seq2seqModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs["tokenizer"]
        super(Seq2seqModel, self).__init__(*args, **kwargs)

    def generate(self, text, out_max_length=40, max_length=512, **kwargs):
        """���������ı������ı�
        [IN]  text: str, �ı�����
              out_max_length: int, ����ı��������
              max_length: int, ���������ı��ܵ������
              **kwargs: beam_search�������
        [OUT] generate_text_list: list[str], �����ı����б�
        """
        # �� һ�� ����������Ӧ�Ľ��
        ## ͨ�������󳤶ȵõ��������󳤶ȣ��������ⲻ�����������󳤶Ȼ���нض�
        # TODO Ӧ����Ϊ���� ����beam_search
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length

        # ������������
        # token_type_id ȫΪ0
        token_ids, token_type_ids = self.tokenizer.encode(text, max_length=input_max_length)
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device).view(1, -1)

        # �����ı�
        generate_list = self.beam_search(
                token_ids,
                token_type_ids,
                self.tokenizer._token_sep_id,
                **kwargs,
                )

        # �����ɵ��ı�����
        generate_text_list = list()
        for cur_output_ids, cur_score in generate_list:
            cur_text = self.tokenizer.decode(cur_output_ids.detach().cpu().numpy())
            generate_text_list.append((cur_text, cur_score.tolist()))

        return generate_text_list

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=None, min_tokens_to_keep=1):
        """beam_searchʱ��top_k_top_p��������������token
        [IN]  logits: tensor, shape=[beam_size, vocab_size], ��ǰ��beam�¸�token���������
              top_k: int, ��������ǰtop_k��token�ĸ��ʣ�����ĸ�������
              top_p: int, �����ɴ�С�����ۼӵ��ճ���top_pʱ�����ǵ�token�������ʣ�����ĸ�������
              filter_value: float, ���������tokenʵ����softmaxǰҪ�ĳɵ�ֵ��Ĭ��-float('Inf')
        """
        # TODO ֱ�ӷŲ���Ĭ��ֵ
        if filter_value is None:
            filter_value = -float('Inf')

        if top_k > 0:
            # ʵ�ʵ�topkҪС��ʵ�ʺ�ѡ��Ŀ
            # ͬʱ��֤����min_tokens_to_keep�ĳ�ȡ���ʲ�Ϊ0
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            # ��Ҫ�����tokenλ��Ϊtrue
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logging.debug("indices_to_remove: {}".format(indices_to_remove))
            # Ϊtrue��λ��ֵ��Ϊfilter_value
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            # �ɴ�С���� ����֮���ۼ� ����������λ�õ�ԭʼind
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            logging.debug("sorted_logits: {}".format(sorted_logits))
            logging.debug("sorted_indices: {}".format(sorted_indices))

            # �ۼ�
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            #cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
            logging.debug("cumulative_probs: {}".format(cumulative_probs))

            # ��Ҫ�����λ��Ϊtrue ���λ����������λ��
            sorted_indices_to_remove = cumulative_probs > top_p
            logging.debug("sorted_indices_to_remove: {}".format(sorted_indices_to_remove))
            if min_tokens_to_keep > 1:
                # ��֤����min_tokens_to_keep�ĳ�ȡ���ʲ�Ϊ0
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            logging.debug("sorted_indices_to_remove: {}".format(sorted_indices_to_remove))
            # ����һλ ��Ϊʹp�պô���top_p����һ��ҲӦ�ñ���
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            logging.debug("sorted_indices_to_remove: {}".format(sorted_indices_to_remove))

            # �����λ�� ��sorted_indicesת��������ǰ�ĸ�λ��
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logging.debug("indices_to_remove: {}".format(indices_to_remove))
            logits[indices_to_remove] = filter_value

        #candidate_num = (logits != filter_value).sum(dim=-1).item()
        #if candidate_num > 1000:
        #    logging.info("candidate num: {}".format(candidate_num))
        return logits

    def add_penalty(self, score, penalty):
        """Ϊscore�ӳͷ�
        [IN]  score: float, ��ǰ�÷�
              penalty: float, Ҫ�ӵĳͷ� ����ֵ�Ĵ�С �в�ͬ�ĳͷ��߼�
        [OUT] score: float, �ͷ����ֵ
        """
        # penaltyΪ0 ��ʾ����
        if penalty != 0:
            if penalty < 1:
                if score < 0:
                    score /= penalty
                else:
                    score *= penalty
            else:
                score -= penalty
        return score

    # stop_idӦ����tokenizer��
    def beam_search(self, token_ids, token_type_ids, stop_id=None,
            beam_size=1, beam_group=1, repeat_penalty=5, diverse_step=5, diverse_penalty=5,
            random_step=1, top_k=0, top_p=1.0, filter_value=None, min_tokens_to_keep=1):
        """beam search����
        [IN]  token_ids: tensor, �����word id
              token_type_ids: tensor, �����segment id
              stop_id: int, ��Ϊֹͣ����token_id��Ĭ����[SEP]��id
              beam_size: int, ָ���Ǹ�beam_group�ڵ�beam_search��֧��С
              beam_group: int, beam_group����
              repeat_penalty: float, ��beam_search��֧������ظ��ַ��ĳͷ�
              diverse_step: int, diverse���ڣ�ÿdiverse_step��ʱ�䲽�Ը�group����diverse�ͷ�
              diverse_penalty: float, ͬʱ�䲽�ڣ���group�����ǰ��group������ַ�һ�µĳͷ�
              random_step: int, ����������ڣ�ÿrandom_step��ʱ�䲽�����������
              top_k: int, �������ʱ�������������ǰtop_k��token����
              top_p: int, �������ʱ��������������ɴ�С�ۻ��͸ճ���top_pʱ���ǵ�token����
              filter_value: int, ��������token�ĸ��ʸ�Ϊfilter_value��softmax����Щtoken�ĸ��ʽ���Ϊ��
              min_tokens_to_keep: int, �������ʱ�����ٱ���min_tokens_to_keep��token���������
        [OUT] generate_list: list[(str, int)], �����ı�����÷ֵ��б�
        """
        if stop_id is None:
            stop_id = self.tokenizer._token_sep_id

        # һ��ֻ����һ��
        # batch_size = 1
        # token_ids shape: [batch_size, seq_length]
        logging.debug("token_ids: {}".format(token_ids))
        logging.debug("token_ids shape: {}".format(token_ids.shape))
        # token_type_ids shape: [batch_size, seq_length]
        logging.debug("token_type_ids : {}".format(token_type_ids))
        logging.debug("token_type_ids  shape: {}".format(token_type_ids.shape))

        # ��ǰΪ��ʼ����beam_size ÿ��ѭ��ʱ �����ִ�beam_group���ı�
        total_beam_size = beam_size * beam_group
        logging.debug("total_beam size: {}".format(total_beam_size))
        repeat_word = [list() for i in range(total_beam_size)]

        # ���������������
        output_ids = torch.empty(1, 0, device=self.device, dtype=torch.long)
        logging.debug("output_ids: {}".format(output_ids))
        logging.debug("output_ids shape: {}".format(output_ids.shape))

        self.model.eval()
        # ��¼���ɵ����м���÷�
        generate_list = list()
        with torch.no_grad():
            # ��ʼ�����÷�
            # output_scores shape: [batch_size]
            output_scores = torch.zeros(token_ids.shape[0], device=self.device)
            # �ظ����� ֱ���ﵽ��󳤶�
            for step in range(self.out_max_length):
                logging.debug("step: {}".format(step))
                #total_beam_size = beam_size * beam_group
                logging.debug("beam size: {}".format(total_beam_size))
                if step == 0:
                    # score shape: [batch_size, seq_length, vocab_size]
                    # TODO ��Ӧ�ø�device����
                    infer_res = self.model(token_ids, token_type_ids, device=self.device)
                    scores = infer_res["token_output"]
                    logging.debug("scores shape: {}".format(scores.shape))
                    # ��һ��ֻ���䵽�� ��Ϊ����Ҫ�Ҹ���topk ���������repeat �ͻ��ظ�ѡ��һ��
                    scores = scores.repeat(beam_group, 1, 1)
                    logging.debug("scores shape: {}".format(scores.shape))

                    # ��һ��Ԥ�������ܸı�
                    # �ظ�beam-size�� ����ids
                    # token_ids shape: [total_beam_size, batch_size*seq_length]
                    token_ids = token_ids.view(1, -1).repeat(total_beam_size, 1)
                    logging.debug("token_ids shape: {}".format(token_ids.shape))

                    # token_type_ids shape: [total_beam_size, batch_size*seq_length]
                    token_type_ids = token_type_ids.view(1, -1).repeat(total_beam_size, 1)
                    logging.debug("token_type_ids shape: {}".format(token_type_ids.shape))
                else:
                    # score shape: [total_beam_size, cur_seq_length, vocab_size]
                    # cur_seq_length���𽥱仯��
                    logging.debug("new_input_ids shape: {}".format(new_input_ids.shape))
                    logging.debug("new_token_type_ids shape: {}".format(new_token_type_ids.shape))
                    # TODO ��Ӧ�ø�device����
                    infer_res = self.model(new_input_ids, new_token_type_ids, device=self.device)
                    scores = infer_res["token_output"]
                    logging.debug("scores shape: {}".format(scores.shape))

                vocab_size = scores.shape[-1]

                # ֻȡ���һ�������vocab�ϵ�score
                # logit_score shape: step0=[beam_group, vocab_size] other=[total_beam_size, vocab_size]
                logit_score = torch.log_softmax(scores[:, -1], dim=-1)
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                # ��ÿһ���ѳ��ֹ���token����score
                if repeat_penalty != 0:
                    for i in range(total_beam_size):
                        for token_id in repeat_word[i]:
                            logit_score[i, token_id] = \
                                    self.add_penalty(logit_score[i, token_id], repeat_penalty)

                # logit_score shape: step0=[beam_group, vocab_size] other=[total_beam_size, vocab_size]
                logit_score = output_scores.view(-1, 1) + logit_score # �ۼƵ÷�
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                # ȡtopk��ʱ������������չƽȻ����ȥ����topk����
                # ͬ��ĸ�beam�����ƽ
                # logit_score shape: step0=[beam_group, vocab_size] other=[beam_group, beam_size*vocab_size]
                logit_score = logit_score.view(beam_group, -1)
                logging.debug("logit_score shape: {}".format(logit_score.shape))

                chosen_beam_inds = list()
                chosen_token_inds = list()
                chosen_scores = list()
                # ��groupҪ���Ⱥ�˳�����beam_search
                # ��Ϊ��group��Ӱ���������group
                previous_output_token = set()
                for group_ind, cur_group_score in enumerate(logit_score):
                    logging.debug("group_ind: {}".format(group_ind))
                    # cur_group_score shape: step0=[vocab_size] other=[beam_size*vocab_size]
                    logging.debug("cur_group_score shape: {}".format(cur_group_score.shape))

                    # ����ǰ��group�����token��ӳͷ�
                    if diverse_penalty > 0 and step % diverse_step == diverse_step - 1:
                        # cur_beam_size step0=1, other=beam_size
                        cur_beam_size = cur_group_score.shape[0] // vocab_size
                        for cur_token_id in previous_output_token:
                            for cur_beam_ind in range(cur_beam_size):
                                cur_group_score[cur_token_id + cur_beam_ind * vocab_size] = self.add_penalty(
                                        cur_group_score[cur_token_id + cur_beam_ind * vocab_size], diverse_penalty)

                    # ��ǰgroup��beam���
                    if random_step > 0 and step % random_step == 0:
                        cur_group_score = self.top_k_top_p_filtering(
                                cur_group_score, top_k, top_p, filter_value, min_tokens_to_keep)

                        cur_group_score = torch.nn.functional.softmax(cur_group_score, dim=-1)
                        logging.debug("cur_group_score: {}".format(cur_group_score))

                        cur_chosen_pos = torch.multinomial(cur_group_score, num_samples=beam_size)

                        # ��cur_group_score��һά���� ���Կ���ֱ����cur_chosen_pos
                        #temp = logits[torch.arange(0, next_tokens.shape[0]).view(-1, 1), cur_chosen_pos]
                        cur_chosen_score = cur_group_score[cur_chosen_pos]
                    else:
                        cur_chosen_score, cur_chosen_pos = torch.topk(cur_group_score, beam_size)

                    logging.debug("cur_chosen_pos: {}".format(cur_chosen_pos))
                    logging.debug("cur_chosen_score: {}".format(cur_chosen_score))
                    logging.debug("cur_chosen_score shape: {}".format(cur_chosen_score.shape))
                    logging.debug("cur_chosen_pos shape: {}".format(cur_chosen_pos.shape))

                    # ��¼��ǰgroupѡ�е�score
                    chosen_scores.append(cur_chosen_score.view(1, -1))

                    # ��¼��ǰgroupѡ�е�beam_ind������������beam_ind��
                    cur_chosen_beam_inds = (cur_chosen_pos // vocab_size) + group_ind * beam_size
                    logging.debug("cur_chosen_beam_inds: {}".format(cur_chosen_beam_inds))
                    chosen_beam_inds.append(cur_chosen_beam_inds)

                    cur_chosen_token_inds = cur_chosen_pos % vocab_size
                    logging.debug("cur_chosen_token_inds: {}".format(cur_chosen_token_inds))
                    chosen_token_inds.append(cur_chosen_token_inds)

                    # ���µ�ǰgroup�����token
                    if diverse_penalty > 0 and step % diverse_step == diverse_step - 1:
                        for token_id in cur_chosen_token_inds:
                            previous_output_token.add(token_id.item())
                    logging.debug("previous_output_token: {}".format(previous_output_token))

                # chosen_beam_inds shape: [total_beam_size]
                chosen_beam_inds = torch.cat(chosen_beam_inds, dim=0)
                logging.debug("chosen_beam_inds: {}".format(chosen_beam_inds))

                # ������Ҫ�ӵ�output_ids�� ����Ҫ���������
                # chosen_token_inds shape: [total_beam_size, 1]
                chosen_token_inds = torch.cat(chosen_token_inds, dim=0).view(-1, 1)
                logging.debug("chosen_token_inds: {}".format(chosen_token_inds))

                # ����� �����Ǹ����top beamsize������ ֮��Ҫ���������߷�
                # chosen_scores shape: [beam_group, beam_size]
                chosen_scores = torch.cat(chosen_scores, dim=0)
                logging.debug("chosen_scores: {}".format(chosen_scores))

                if repeat_penalty != 0:
                    # ��Ҫ���µ�repeat_word������ ��Ȼ���beam��������ͬһ�������
                    # �ڶ�������ʱ ��ѵ�һ�ε�Ҳ���ڵڶ�����
                    new_repeat_word = [list() for i in range(total_beam_size)]
                    logging.debug("repeat_word: {}".format(repeat_word))
                    for index, (beam_ind, word_ind) in enumerate(zip(chosen_beam_inds, chosen_token_inds)):
                        logging.debug("beam_ind: {}".format(beam_ind))
                        logging.debug("word_ind: {}".format(word_ind))
                        new_repeat_word[index] = repeat_word[beam_ind].copy()
                        new_repeat_word[index].append(word_ind.item())
                    repeat_word = new_repeat_word
                    logging.debug("repeat_word: {}".format(repeat_word))

                # ���µ÷�
                # output_scores shape: [total_beam_size]
                output_scores = chosen_scores.view(-1)
                logging.debug("output_scores: {}".format(output_scores))

                # ����output_ids
                # ͨ��chosen_beam_indsѡ���ĸ�beam
                # ͨ��chosen_token_indsѡ��ǰbeam���ĸ�token_id
                # output_ids shape: [total_beam_size, cur_seq_length]
                output_ids = torch.cat([output_ids[chosen_beam_inds], chosen_token_inds], dim=1).long()
                logging.debug("output_ids: {}".format(output_ids))

                # new_input_ids shape: [total_beam_size, cur_seq_length]
                # token_ids�ǹ̶�ԭ����
                # output_ids�ǵ�ǰbeam_search���µ�total_beam_size����ѡ·��
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                logging.debug("new_input_ids shape: {}".format(new_input_ids.shape))

                # new_token_type_ids shape: [total_beam_size, cur_seq_length]
                # token_type_ids��ӵ�typeȫΪ1
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                logging.debug("new_token_type_ids shape: {}".format(new_token_type_ids.shape))

                # ��¼��ǰoutput_ids����sep_id�����
                end_counts = (output_ids == stop_id).sum(dim=1)  # ͳ�Ƴ��ֵ�end���
                logging.debug("end_counts: {}".format(end_counts))
                logging.debug("end_counts shape: {}".format(end_counts.shape))
                assert (end_counts < 2).all(), "wrong end_counts: {}".format(end_counts)

                best_beam_inds = chosen_scores.argmax(dim=1)
                logging.debug("best_beam_inds: {}".format(best_beam_inds))
                best_beam_inds += torch.tensor(range(beam_group), device=self.device).view(-1) * beam_size
                logging.debug("best_beam_inds: {}".format(best_beam_inds))

                # ��¼��beam�Ƿ���Ҫ����
                continue_flag = [True for _ in range(total_beam_size)]

                # �����ѽ���������
                for group_ind in range(beam_group):
                    for beam_ind in range(beam_size):
                        cur_ind = group_ind * beam_size + beam_ind
                        # ֻ�н�����beam��Ҫ����
                        if end_counts[cur_ind] > 0:
                            # �����ǰ������beamͬʱҲ������score��ߵ� ��������
                            cur_best_ind = best_beam_inds[group_ind]
                            if cur_ind == cur_best_ind:
                                logging.debug("cur_best: {}".format(cur_best_ind))
                                # ������list
                                generate_list.append((
                                    output_ids[cur_best_ind],
                                    output_scores[cur_best_ind].detach().cpu().numpy(),
                                    ))
                                # beam_group�����޸� ����Ӱ��range(beam_group)
                                # beam_group����һ
                                beam_group -= 1
                                # ��ǰ��ֹͣ
                                for i in range(beam_size * group_ind, beam_size * (group_ind + 1)):
                                    continue_flag[i] = False
                            else:
                                # �����beam�ѽ��� ������������߷� ��������߷�copy����ǰ������beam��
                                # ���beam��token_ids��token_type_ids��һ�� �������ﲻ��Ҫcopy
                                new_input_ids[cur_ind] = new_input_ids[cur_best_ind].clone()
                                new_token_type_ids[cur_ind] = new_token_type_ids[cur_best_ind].clone()
                                output_ids[cur_ind] = output_ids[cur_best_ind].clone()
                                output_scores[cur_ind] = output_scores[cur_best_ind].clone()
                                # repeat_word��Ԫ����list ��copy
                                repeat_word[cur_ind] = repeat_word[cur_best_ind].copy()

                logging.debug("continue_flag: {}".format(continue_flag))

                # ȥ������ɵ�����
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

                # ��һ�ֵ�����total_beam_size����
                total_beam_size = beam_size * beam_group
                logging.debug("total beam size: {}".format(total_beam_size))

                # �������� ��total_beam_sizeΪ0 ��beam_groupΪ0ʱ ����
                if total_beam_size == 0:
                    break

            # generate_list��������
            generate_list = sorted(generate_list, key=lambda x:x[1], reverse=True)
            return generate_list


class BertSeq2seqModel(Seq2seqModel):
    def __init__(self, *args, **kwargs):
        """��ʼ��
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
        # TODO Seq2seqModelӦ��û��label_encoder
        self.label_encoder = kwargs.pop("label_encoder", None)
        return super(BertSeq2seqModel, self).train(*args, **kwargs)

    # ���ܿ��Էŵ�Seq2seqModel��
    def eval(self, eval_dataloader, print_step=50, gather_loss=True, **kwargs):
        self.model.eval()
        cur_eval_step = 0
        start_time = time.time()
        loss_list = list()
        # ��֤ʱ�����淴����ݶ�
        with torch.no_grad():
            for batch in eval_dataloader:
                cur_eval_step += 1
                loss = self.get_loss(**batch)
                # ����lossʱ �Ƚ���detach ��Ȼ����Ĳ�ֻ��loss ������������ͼ
                loss_list.append(loss.detach().item())
                if cur_eval_step % print_step == 0:
                    cost_time = time.time() - start_time
                    speed = cur_eval_step / cost_time
                    logging.info('infer step %d, total cost time = %.4fs, speed %.2f step/s' \
                            % (cur_eval_step, cost_time, speed))
        loss_mean = np.mean(loss_list)
        if self.distributed:
            # ���ֲ�ʽѵ��ʱ ���Ҫ����ȫ����loss
            # �����²���
            if gather_loss:
                loss_tensor = torch.tensor(loss_mean).to(self.device)
                # ����ֻ��ӡmaster���̵�loss ����ֻ��Ҫreduce��rankΪ0�Ľ���
                # ���Ҫ���н���loss_tensorͬ�� ��all_reduce
                torch.distributed.reduce(loss_tensor, 0, op=torch.distributed.ReduceOp.SUM)
                if self.is_master:
                    logging.debug("rank {} gather loss total= {}.".format(self.local_rank, loss_tensor))
                    loss_mean = loss_tensor / torch.distributed.get_world_size()
                    logging.info("rank {} gather loss = {}.".format(self.local_rank, loss_mean))
            elif self.is_master:
                # ����ֻ��master���̴�ӡloss
                logging.info("rank {} local loss = {}.".format(self.local_rank, loss_mean))
        else:
            logging.info("eval loss = {}.".format(loss_mean))

        return loss_mean

    def check_if_best(self, cur_eval_res):
        """������������ж��Ƿ�����
        [IN]  cur_eval_res: float, ��ǰ�����÷�
        [OUT] true��Ϊ��ǰ���ŵ÷֣�������
        """
        if self.min_loss is None or self.min_loss >= cur_eval_res:
            self.min_loss = cur_eval_res
            return True
        else:
            return False

    def get_best_score(self):
        """���ص�ǰ���ŵ÷�
        """
        return self.min_loss
