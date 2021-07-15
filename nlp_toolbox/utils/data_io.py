#!/usr/bin/env python
# -*- coding: gb18030 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_io.py
Author: zhanghao55(zhanghao55@baidu.com)
Date: 2019/09/19 20:44:30
"""

import codecs
from collections import defaultdict
from collections import namedtuple
import logging
import numpy as np
#import paddle.fluid.dygraph as D
import pickle
import os
import time

from sklearn.datasets import dump_svmlight_file


def get_data(data_path, read_func=lambda x:x, header=False, encoding="gb18030", verbose=False):
    """��ȡ���ļ�(��Ŀ¼�������ļ�)������
    [in]  data_path : str, ���ݼ���ַ
          verbose   : bool, �Ƿ�չʾ������Ϣ
    [out] data : list[str], ���ļ�(��Ŀ¼�������ļ�)������
    """
    file_list = get_file_name_list(data_path, verbose)
    for file_index, file_path in enumerate(file_list):
        logging.info("get data from file: {}".format(file_path))
        for line_index, line in enumerate(read_from_file(file_path, read_func, encoding)):
            if header and file_index != 0 and line_index == 0:
                # ����б�ͷ �����һ���ļ��� ÿ���ļ��ĵ�һ��ʡ��
                continue
            yield line


def get_data_with_header(data_path, sep="\t", encoding="gb18030", verbose=False):
    """��ȡ���ļ�(��Ŀ¼�������ļ�)������
    [in]  data_path : str, ���ݼ���ַ
          verbose   : bool, �Ƿ�չʾ������Ϣ
    [out] data : list[str], ���ļ�(��Ŀ¼�������ļ�)������
    """
    file_list = get_file_name_list(data_path, verbose)
    for file_index, file_path in enumerate(file_list):
        logging.info("get data from file: {}".format(file_path))
        for line_index, line in enumerate(read_from_file(file_path, encoding=encoding)):
            parts = line.rstrip("\n").split(sep)
            if line_index == 0:
                # ÿ���ļ��ĳ�ʼ����ͷ
                Record = namedtuple("record", parts)
            else:
                yield Record(*parts)


def get_file_name_list(data_path, verbose=True):
    """���ɹ������ݼ����ļ��б�
        ������ݼ���ַ���ļ����򷵻��б���ֻ�и��ļ���ַ
        ������ݼ���ַ��Ŀ¼���򷵻��б��а�����Ŀ¼�������ļ�����(����'.'��ͷ���ļ�)
    [in]  data_path : str, ���ݼ���ַ
          verbose   : bool, �Ƿ�չʾ������Ϣ
    [out] file_list : list[str], ���ݼ��ļ������б�
    """
    from collections import deque
    file_list = list()
    path_stack = deque()
    path_stack.append(data_path)
    while len(path_stack) != 0:
        cur_path = path_stack.pop()
        if verbose:
            logging.debug("check data path: %s" % cur_path)
        # ���ȼ��ԭʼ�������ļ������ļ���
        if os.path.isdir(cur_path):
            #logging.debug("data path is directory.")
            files = os.listdir(cur_path)
            # �����ļ����е�ÿһ���ļ�
            for file_name in files:
                # ����ļ�����.��ͷ ˵�����ļ����� ���������������ļ�
                if len(file_name) == 0 or file_name[0] == ".":
                    continue
                file_path = os.path.join(cur_path, file_name)
                path_stack.append(file_path)
        elif os.path.isfile(cur_path):
            #logging.info("data path is file. add to list.")
            file_list.append(cur_path)
        else:
            raise TypeError("unknown type of data path : %s" % cur_path)
    if verbose:
        logging.debug("file list top 20:")
        for index, file_name in enumerate(file_list[:20]):
            logging.debug("#%d: %s" % (index + 1, file_name))
    return file_list


def get_column_values(data_dir, fetch_list, sep="\t", encoding="gb18030"):
    """����ָ���е�����
    [in]  data_dir: str, ���ݼ���ַ
          fetch_list: list[str], ָ���е�����
    [out] res_list: list[list[str]], ��ָ���е������б�
    """
    line_ite = get_data(
            data_dir,
            encoding=encoding,
            )

    res_list = list()
    for _ in fetch_list:
        res_list.append(list())

    for cur_line in line_ite:
        cur_parts = cur_line.split(sep)
        for cur_res_index, cur_column in enumerate(fetch_list):
            res_list[cur_res_index].append(cur_parts[cur_column])

    return res_list


def get_attr_values(data_dir, fetch_list, encoding="gb18030"):
    """���ش��ֶ����������У�ָ���ֶε�����
    [in]  data_dir: str, ���ݼ���ַ
          fetch_list: list[str], ָ�����ֶ����б�
    [out] res_list: list[list[str]], ��ָ�����ֶ����������б�
    """
    record_ite = get_data_with_header(
            data_dir,
            encoding=encoding,
            )

    res_dict = defaultdict(list)
    for cur_record in record_ite:
        for attr_name in fetch_list:
            res_dict[attr_name].append(getattr(cur_record, attr_name))

    res_list = list()
    for attr_name in fetch_list:
        res_list.append(res_dict[attr_name])

    return res_list


def read_from_file(file_path, read_func=lambda x:x, encoding="gb18030"):
    """�����ļ��еĴ�
    [in] file_path: str, �ļ���ַ
    [out] word_list: list[str], �����б�
    """
    with codecs.open(file_path, "r", encoding) as rf:
        for line in rf:
            res = read_func(line.strip("\n"))
            if res is not None:
                yield res


def write_to_file(text_list, dst_file_path, write_func=lambda x:x, encoding="gb18030"):
    """���ı��б����Ŀ���ļ���ַ
    [in]  text_list: list[str], �ı��б�
          dst_file_path: str, Ŀ���ļ���ַ
    """
    with codecs.open(dst_file_path, "w", encoding) as wf:
        # ����ֱ��ȫ��join ��Щ���ݹ��� Ӧ��for
        #wf.write("\n".join([write_func(x) for x in text_list]))
        for text in text_list:
            if text is None:
                continue
            res = write_func(text)
            if res is None:
                continue
            wf.write(res + "\n")


def load_pkl(pkl_path):
    """���ض���
    [in]  pkl_path: str, �����ļ���ַ
    [out] obj: class, ����
    """
    logging.debug("load from \"{}\".".format(pkl_path))
    start_time = time.time()
    with open(pkl_path, 'rb') as rf:
        return pickle.load(rf)
    logging.debug("load from \"{}\" succeed, cost time: {:.2f}s.".format(pkl_path, time.time() - start_time))


def dump_pkl(obj, pkl_path, overwrite=False):
    """�洢����
    [in]  obj: class, ����
          pkl_path: str, �����ļ���ַ
          overwrite: bool, �Ƿ񸲸ǣ�False���ļ�����ʱ���洢
    """
    if len(pkl_path) == 0 or pkl_path is None:
        logging.warning("pkl_path(\"%s\") illegal." % pkl_path)
    elif os.path.exists(pkl_path) and not overwrite:
        logging.warning("pkl_path(\"%s\") already exist and not over write." % pkl_path)
    else:
        with open(pkl_path, 'wb') as wf:
            pickle.dump(obj, wf)
        logging.debug("save to \"%s\" succeed." % pkl_path)


def label_encoder_save_as_class_id(label_encoder, class_id_path, conf_thres=0.5):
    """��LabelEncoder����תΪdef-user�е�class_id.txt��ʽ����ʽ����ָ���ļ�
    [in]  label_encoder: class, ����
          class_id_path: str, �洢�ļ���ַ
          conf_thres: float, ������ֵ ����ֻ��ͳһ����
    """
    class_id_list = ["%d\t%s\t%f" % (index, str(class_name), conf_thres) for \
            index, class_name in enumerate(label_encoder.classes_)]
    write_to_file(class_id_list, class_id_path)
    logging.debug("trans label_encoder to \"%s\" succeed." % class_id_path)


def dump_libsvm_file(X, y, file_path, zero_based=False):
    """�����ݼ�תΪlibsvm��ʽ liblinear��xgboost��lightgbm�����Խ��ոø�ʽ
    [in]  X: array-like��sparse matrix, ��������
          y: array-like��sparse matrix, �����
          file_path: string��file-like in binary model, �ļ���ַ�����߶�������ʽ�򿪵Ŀ�д�ļ�
          zero_based: bool, true������id��0��ʼ liblinearѵ��ʱҪ������id��1��ʼ ���һ����ҪΪFalse
    """
    logging.debug("trans libsvm format data to %s." % file_path)
    start_time = time.time()
    dump_svmlight_file(X, y, file_path, zero_based=zero_based)
    logging.info("cost_time : %.4fs" % (time.time() - start_time))


#def load_model(init_model, model_path):
#    """ ��ѵ���õ��Ĳ������ص�paddle��̬ͼģ�ͽṹ��
#    [in] init_model: �ѹ���õ�ģ�ͽṹ
#         model_path: str, ģ�͵�ַ(ȥ��.pdparams��׺)
#    """
#    if os.path.exists(model_path + ".pdparams"):
#        logging.info("load model from {}".format(model_path))
#        start_time = time.time()
#        sd, _ = D.load_dygraph(model_path)
#        init_model.set_dict(sd)
#        logging.info("cost time: %.4fs" % (time.time() - start_time))
#    else:
#        logging.info("cannot find model file: {}".format(model_path + ".pdparams"))


def gen_batch_data(data_iter, batch_size=32, max_seq_len=300, max_ensure=False, with_label=True):
    """ ����������
    [IN] data_iter: iterable, �ɵ���������
         batch_size: int, ����С
         max_seq_len: int, ������󳤶�
         max_ensure: bool, True��̶���󳤶ȣ����򰴸�����󳤶���padding
         with_label: �������Ƿ���label��label����padding��
    """
    batch_data = list()

    def pad(data_list):
        """ padding
        [IN]  data_list: ��padding������
        [OUT] padding�õ�����
        """
        # ��������
        # ȷ����ǰ������󳤶�
        if max_ensure:
            cur_max_len = max_seq_len
        else:
            cur_max_len = max([len(x) for x in data_list])
            cur_max_len = max_seq_len if cur_max_len > max_seq_len else cur_max_len

        # padding
        return [np.pad(x[:cur_max_len], [0, cur_max_len - len(x[:cur_max_len])], mode='constant') for x in data_list]

    def batch_process(cur_batch_data, cur_batch_size):
        """ ���ɶ������ݽ���padding����
        [IN]  cur_batch_data: list(list), ��������
              cur_batch_size: int, ������С
        [OUT] padding�õ�������
        """
        batch_list = list()
        data_lists = zip(*cur_batch_data)
        #print("cur batch_size = {}".format(cur_batch_size))
        if with_label:
            label_list = data_lists[-1]
            data_lists = data_lists[:-1]

        for data_list in data_lists:
            #print(data_list)
            data_list = pad(data_list)
            data_np = np.array(data_list).reshape([cur_batch_size, -1])
            batch_list.append(data_np)

        if with_label:
            label_np = np.array(label_list).reshape([cur_batch_size, -1])
            batch_list.append(label_np)

        return batch_list

    for data in data_iter:
        if len(batch_data) == batch_size:
            # ��ǰ�����һ��batch
            yield batch_process(batch_data, batch_size)
            batch_data = list()
        batch_data.append(data)

    if len(batch_data) > 0:
        yield batch_process(batch_data, len(batch_data))


if __name__ == "__main__":
    pass
