#!/usr/bin/env python3
# -*- coding:gb18030 -*-
"""
File  :   sampler.py
Author:   zhanghao55@baidu.com
Date  :   21/01/22 15:32:26
Desc  :   
"""

import sys
import random
import heapq


class Sampler(object):
    """通用抽样类 蓄水池抽样 支持加权抽样 去重抽样
    """
    def __init__(self, sample_num, uniq=False):
        """初始化 指定抽样个数
        Args:
            sample_num : 指定抽样个数
            uniq       : True则抽样结果去重
        Caution:
            当uniq为True时，要保证插入的key可hash
        """
        assert isinstance(sample_num, int), "sample_num must be integer."
        self._sample_num = sample_num
        self._sample_heap = list()
        self._uniq = uniq
        if self._uniq:
            # 记录在抽样数据集中的obj的key集合
            self._uniq_key_set = set()

    def put(self, obj, weight=1.0, key=None):
        """
        Args:
            obj : 抽样的元素
            weight: 当前元素抽样权重
            key : 去重时依赖的键，如果为None，则obj为键
        Return:
            True表示插入成功，False表示插入失败(当要去重，且数据已在抽样结果中时插入失败)
        Caution:
            当uniq为True时，要保证插入的obj可hash
        """
        # 如果要去重 需要确认是否已重复
        if self._uniq:
            # 确定该数据的uniq_key
            uniq_key = obj if key is None else key

            # 如果已存在 则不插入
            if uniq_key in self._uniq_key_set:
                # 且该obj在_uniq_key_set中已存在 也不需要插入
                return False

            insert_element = (uniq_key, obj)
        else:
            insert_element = obj

        # 为当前数生成抽样权重
        # 随机数 in [0, 1)
        # 蓄水池加权抽样
        # 参考:《Weighted Random Sampling over Data Streams》
        # https://arxiv.org/pdf/1012.0256.pdf
        rand_R = random.uniform(0, 1)
        sample_weight = rand_R ** (1 / float(weight))

        # 如果抽样权重满足条件 则添加
        if len(self._sample_heap) < self._sample_num or \
                sample_weight > self._sample_heap[0][0]:
            heapq.heappush(self._sample_heap, (sample_weight, insert_element))
            if self._uniq:
                # 添加新插入数据的记录
                self._uniq_key_set.add(uniq_key)

        # 抽样数据量大于指定数时 弹出多余数据
        # 如果要去重 则需要更新重复数据记录
        if len(self._sample_heap) > self._sample_num:
            _, replaced_element = heapq.heappop(self._sample_heap)
            if self._uniq:
                replaced_key, _ = replaced_element
                self._uniq_key_set.remove(replaced_key)

        return True

    def get_sample_list(self):
        """返回当前抽样结果
        """
        if self._uniq:
            # 需要去重时 抽样元素是二元组 需要的只有第二个元素
            return [x[1][1] for x in self._sample_heap]
        else:
            return [x[1] for x in self._sample_heap]

    def clear(self):
        """清空抽样列表
        """
        self._sample_heap[:]=[]
        if self._uniq:
            self._uniq_key_set.clear()


def test():
    """测试基本功能
    """

    sampler = Sampler(4)
    for _ in range(10):
        sampler.clear()
        for num in range(1, 12):
            sampler.put(num)
        print(sampler.get_sample_list())

    sampler = Sampler(10)
    for i in range(100):
        sampler.put(i)
    print(sampler.get_sample_list())

    print("抽样结果不去重")
    sampler = Sampler(10)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            sampler.put(random.randint(0, 5))
        print(sampler.get_sample_list())

    print("抽样结果去重")
    sampler = Sampler(10, True)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            sampler.put(random.randint(0, 5))
        print(sampler.get_sample_list())

    print("抽样结果去重 key没有重复")
    sampler = Sampler(10, True)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            sampler.put(random.randint(0, 5), key=i)
        print(sampler.get_sample_list())

    print("抽样结果去重 key重复")
    sampler = Sampler(10, True)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            cur_v = random.randint(0, 5)
            sampler.put(cur_v, key=cur_v)
        print(sampler.get_sample_list())

    print("抽样结果不去重 key重复")
    sampler = Sampler(10, False)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            cur_v = random.randint(0, 5)
            sampler.put(cur_v, key=cur_v)
        print(sampler.get_sample_list())

    print("抽样结果不去重 key不重复 按权值去重")
    from collections import defaultdict
    sampler = Sampler(1, False)
    count_dict = defaultdict(int)
    weight_dict = {
            0: 0.6,
            1: 0.3,
            2: 0.1,
            }
    for _ in range(100000):
        sampler.clear()
        for i in range(3):
            sampler.put(i, weight=weight_dict[i])
        sample_res = sampler.get_sample_list()[0]
        count_dict[sample_res] += 1

    print("count res: {}".format(count_dict))


if __name__ == '__main__': 
    test()

