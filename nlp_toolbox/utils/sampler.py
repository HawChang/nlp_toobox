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
    """ͨ�ó����� ��ˮ�س��� ֧�ּ�Ȩ���� ȥ�س���
    """
    def __init__(self, sample_num, uniq=False):
        """��ʼ�� ָ����������
        Args:
            sample_num : ָ����������
            uniq       : True��������ȥ��
        Caution:
            ��uniqΪTrueʱ��Ҫ��֤�����key��hash
        """
        assert isinstance(sample_num, int), "sample_num must be integer."
        self._sample_num = sample_num
        self._sample_heap = list()
        self._uniq = uniq
        if self._uniq:
            # ��¼�ڳ������ݼ��е�obj��key����
            self._uniq_key_set = set()

    def put(self, obj, weight=1.0, key=None):
        """
        Args:
            obj : ������Ԫ��
            weight: ��ǰԪ�س���Ȩ��
            key : ȥ��ʱ�����ļ������ΪNone����objΪ��
        Return:
            True��ʾ����ɹ���False��ʾ����ʧ��(��Ҫȥ�أ����������ڳ��������ʱ����ʧ��)
        Caution:
            ��uniqΪTrueʱ��Ҫ��֤�����obj��hash
        """
        # ���Ҫȥ�� ��Ҫȷ���Ƿ����ظ�
        if self._uniq:
            # ȷ�������ݵ�uniq_key
            uniq_key = obj if key is None else key

            # ����Ѵ��� �򲻲���
            if uniq_key in self._uniq_key_set:
                # �Ҹ�obj��_uniq_key_set���Ѵ��� Ҳ����Ҫ����
                return False

            insert_element = (uniq_key, obj)
        else:
            insert_element = obj

        # Ϊ��ǰ�����ɳ���Ȩ��
        # ����� in [0, 1)
        # ��ˮ�ؼ�Ȩ����
        # �ο�:��Weighted Random Sampling over Data Streams��
        # https://arxiv.org/pdf/1012.0256.pdf
        rand_R = random.uniform(0, 1)
        sample_weight = rand_R ** (1 / float(weight))

        # �������Ȩ���������� �����
        if len(self._sample_heap) < self._sample_num or \
                sample_weight > self._sample_heap[0][0]:
            heapq.heappush(self._sample_heap, (sample_weight, insert_element))
            if self._uniq:
                # ����²������ݵļ�¼
                self._uniq_key_set.add(uniq_key)

        # ��������������ָ����ʱ ������������
        # ���Ҫȥ�� ����Ҫ�����ظ����ݼ�¼
        if len(self._sample_heap) > self._sample_num:
            _, replaced_element = heapq.heappop(self._sample_heap)
            if self._uniq:
                replaced_key, _ = replaced_element
                self._uniq_key_set.remove(replaced_key)

        return True

    def get_sample_list(self):
        """���ص�ǰ�������
        """
        if self._uniq:
            # ��Ҫȥ��ʱ ����Ԫ���Ƕ�Ԫ�� ��Ҫ��ֻ�еڶ���Ԫ��
            return [x[1][1] for x in self._sample_heap]
        else:
            return [x[1] for x in self._sample_heap]

    def clear(self):
        """��ճ����б�
        """
        self._sample_heap[:]=[]
        if self._uniq:
            self._uniq_key_set.clear()


def test():
    """���Ի�������
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

    print("���������ȥ��")
    sampler = Sampler(10)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            sampler.put(random.randint(0, 5))
        print(sampler.get_sample_list())

    print("�������ȥ��")
    sampler = Sampler(10, True)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            sampler.put(random.randint(0, 5))
        print(sampler.get_sample_list())

    print("�������ȥ�� keyû���ظ�")
    sampler = Sampler(10, True)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            sampler.put(random.randint(0, 5), key=i)
        print(sampler.get_sample_list())

    print("�������ȥ�� key�ظ�")
    sampler = Sampler(10, True)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            cur_v = random.randint(0, 5)
            sampler.put(cur_v, key=cur_v)
        print(sampler.get_sample_list())

    print("���������ȥ�� key�ظ�")
    sampler = Sampler(10, False)
    for _ in range(10):
        sampler.clear()
        for i in range(100):
            cur_v = random.randint(0, 5)
            sampler.put(cur_v, key=cur_v)
        print(sampler.get_sample_list())

    print("���������ȥ�� key���ظ� ��Ȩֵȥ��")
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

