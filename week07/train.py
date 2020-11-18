#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 下午9:37
# @Author  : lovemefan
# @File    : train.py

import argparse
from tqdm import trange, tqdm
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
    parser.add_argument('--learning_rate', help='learning_rate', default=2e-6)
    args = parser.parse_args()
    return args





if __name__ == '__main__':
    # 初始化模型中所有的参数
    np.zeros(1)
    args = init_args()
    # 初始化模型参数
    model = Model(args)
    dataLoader = DataLoader()
    batch_generator = dataLoader.get_train_batch(int(args.batch_size))
    epoch_size = 1

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for img_batch, label_batch in batch_generator:
        for index in trange(len(img_batch),desc = "batch"):
            model.train_batch(img_batch, label_batch, args.learning_rate)
            # 训练集平均loss
            train_loss_temp = model.loss_function(img_batch, label_batch) / len(label_batch)
            train_loss.append(train_loss_temp)
            # 测试集平均loss
            test_loss_temp = model.loss_function(dataLoader.test_img[index:index+len(label_batch)],
                                                 dataLoader.test_label[index:index+len(label_batch)]) / len(label_batch)
            test_loss.append(test_loss_temp)

        # 用训练集检验准确率
        train_data = (dataLoader.test_img[:1000], dataLoader.test_label[:1000])
        train_data_size = len(train_data[1])
        tmp = [model.predict_label(train_data[0][i]).argmax() == train_data[1][i] for i in range(train_data_size)]
        train_accuracy = tmp.count(True) / len(tmp)

        # 利用测试集检验准确率
        test_data = (dataLoader.test_img, dataLoader.test_label)
        test_data_size = len(test_data[1])
        tmp = [model.predict_label(test_data[0][i]).argmax() == test_data[1][i] for i in range(test_data_size)]
        test_accuracy = tmp.count(True)/len(tmp)

        print(f"train_loss:{train_loss_temp}")
        print(f"test_loss:{test_loss_temp}")
        print(f"train_accuracy:{train_accuracy}")
        print(f"test_accuracy:{test_accuracy}")


