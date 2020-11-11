#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 下午9:37
# @Author  : lovemefan
# @File    : train.py

import argparse
import math

import numpy as np
def init_args():
    parser = argparse.ArgumentParser(description="输入参数")
    parser.add_argument('--input_dim', help='dimension of input')
    parser.add_argument('--hidden_dim', help='dimension of hidden layer')
    parser.add_argument('--output_dim', help='dimension of output')
    args = parser.parse_args()
    return args


def tanh(x):
    """tanh(x) = (e^x - e^-x)/(e^x + e^-x)"""
    return  np.tanh(x)


def softmax(x):
    """softmax(x) = e^(x_i)/sum(e^(x_i))"""
    exp = np.exp(x - x.max())
    return exp / exp.sum()

def init_parmeters_b(layer):
    """初始化参数b
    :param layer 初始化第几层的参数
    """
    dist = distributuion[layer]['b']
    return np.random.rand(dimensions[layer])

def init_parmenters_w(layer):
    pass

arg = init_args()
# 激活函数
activation = [tanh, softmax]

dimensions = [int(arg.input_dim), int(arg.output_dim)]
#
distributuion = [
    # 第一层,初始化b的取值范围
    {'b': [0, 0]},
    {'b': [0, 0], 'w': [-math.sqrt(6/(dimensions[0] + dimensions[1])), math.sqrt(6/(dimensions[0] + dimensions[1]))]}
]
if __name__ == '__main__':


    x = np.array([1, 2, 3, -1])
    print(tanh(0.1))
    print(softmax(x))
    print(distributuion)