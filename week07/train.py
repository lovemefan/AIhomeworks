#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 下午9:37
# @Author  : lovemefan
# @File    : train.py

import argparse
import math
import numpy as np

from week07.dataLoader import DataLoader
from week07.model import Model


def init_args():
    """初始化输入参数"""
    parser = argparse.ArgumentParser(description="输入参数")
    parser.add_argument('--input_dim', help='dimension of input', default=784)
    parser.add_argument('--hidden_dim', help='dimension of hidden layer', default=1024)
    parser.add_argument('--output_dim', help='dimension of output', default=10)
    parser.add_argument('--batch_size', help='batch_size', default=64)
    parser.add_argument('--learning_rate', help='learning_rate', default=0.01)
    args = parser.parse_args()
    return args





if __name__ == '__main__':
    # 初始化模型中所有的参数
    args = init_args()
    # 初始化模型参数
    model = Model(args)
    dataLoader = DataLoader()
    img = dataLoader.train_img[0]
    label = dataLoader.train_label[0]
    print(f"label：{label}")
    predict = model.predict_label(dataLoader.train_img[0])
    print(f"predict：{predict}")
    model.grad_parameters(img,label)