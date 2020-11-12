#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 下午9:37
# @Author  : lovemefan
# @File    : train.py

import argparse
import math
import numpy as np

from week07.model import Model


def init_args():
    """初始化输入参数"""
    parser = argparse.ArgumentParser(description="输入参数")
    parser.add_argument('--input_dim', help='dimension of input')
    parser.add_argument('--hidden_dim', help='dimension of hidden layer')
    parser.add_argument('--output_dim', help='dimension of output')
    args = parser.parse_args()
    return args









if __name__ == '__main__':
    # 初始化模型中所有的参数
    args = init_args()
    # 初始化模型参数
    model = Model(args)
    parameters = model.parameters
    predict = predict_label(np.random.rand(784), parameters)
    print(predict)
    pass