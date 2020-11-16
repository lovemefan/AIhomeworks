#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 下午2:53
# @Author  : lovemefan
# @File    : model.py
import math
import numpy as np


class Model:
    """
    模型使用三层全连接神经网络
    输入为 28*28=784 的图片
    隐藏层个数为1024，参数为 784*1024
    输出为10个分类
    """
    def __init__(self, args, parameters=None):
        """初始化模型参数
        :param arg 为可控制台输入的超参数
        :param parameters 指定参数，可用于训练完成后的参数
        """
        self.args = args
        # 储存维度信息
        self.dimensions = [int(args.input_dim), int(args.hidden_dim), int(args.output_dim)]
        # 模型结构
        self.distribution = [
            # 第一层,初始化b的取值，[0,0] 表示初始化b的范围为[0,0]
            {'b': [0, 0]},
            # 第二层,初始化w，b的取值
            {'b': [0, 0], 'w': [-math.sqrt(6 / (self.dimensions[0] + self.dimensions[1])),
                                math.sqrt(6 / (self.dimensions[0] + self.dimensions[1]))]},
            {'b': [0, 0], 'w': [-math.sqrt(6 / (self.dimensions[1] + self.dimensions[2])),
                                math.sqrt(6 / (self.dimensions[1] + self.dimensions[2]))]}
        ]
        # 模型中用到的激活函数
        self.activetion = [self.tanh, self.tanh, self.softmax]
        # 存储激活函数对应的导数
        self.diffential = {self.tanh:self.d_tanh,
                             self.softmax:self.d_softmax,
                             self.sigmoid:self.sigmoid}

        # 如果没指定参数随机初始化
        if parameters:
            self.parameters = parameters
        else:
            self.parameters = self.init_parameters()

    def tanh(self, x):
        """tanh激活函数
        :return tanh(x) = (e^x - e^-x)/(e^x + e^-x)"""
        return np.tanh(x)

    def d_tanh(self, x):
        """tanh的导数
        """
        return 1 - np.power(self.tanh(x), 2)

    def softmax(self, x):
        """
        softmax(x) = e^(x_i)/sum(e^(x_i))"""
        exp = np.exp(x - x.max())
        return exp / exp.sum()

    def d_softmax(self, x):
        """softmax对每个变量的导数
        返回一个二维矩阵，第i行表示对第i个求导
        """
        mm = self.softmax(x)
        return np.diag(mm) * np.outer(mm, mm)

    def sigmoid(self, x):
        """sigmoid 激活函数
        :return sigmoid(x) = 1/（1 + e^-x ）
        """
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        """sigmoid的导数"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def init_parameter(self, layer, parameter):
        """初始化一个参数
        :param layer 初始化第几层的参数
        :param parmeter 参数，b或w
        :return 返回初始化后的矩阵
        """
        dist = self.distribution[layer][parameter]
        # 随机初始化，范围为dist[0]到dist[1]
        if parameter == 'b':
            # 返回一个一维矩阵
            return np.random.rand(self.dimensions[layer]) * (dist[1] - dist[0]) + dist[0]
        elif parameter == 'w':
            # 返回一个二维矩阵
            return np.random.rand(self.dimensions[layer - 1],
                                  self.dimensions[layer]) * (dist[1] - dist[0]) + dist[0]

    def init_parameters(self):
        """初始化模型所有参数"""

        # 储存每一层的参数
        self.parameters = []
        # 初始化每一层的参数
        for i in range(len(self.distribution)):
            layer_parameter = {}
            for key, value in self.distribution[i].items():
                layer_parameter[key] = self.init_parameter(i, key)

            self.parameters.append(layer_parameter)
        return self.parameters

    def onehot_encode(self, label):
        """将一个数字编码成onehot矩阵"""
        one_hot = np.zeros(int(self.args.output_dim))
        one_hot[label] = 1
        return one_hot

    def loss_function(self, img, label):
        """这里的损失函数取二范数
        :param img 输入数据
        :param label 输入的标签
        :param parameters 模型的参数
        """
        y_pred = self.predict_label(img, self.parameters)
        y = self.onehot_encode(label)
        diff = y - y_pred
        return np.dot(diff, diff)

    def grad_parameters(self, img, label):
        """计算模型参数的梯度
        :param img 模型输入，即图片像素拉成的的一维的矩阵
        :param label 输入样本的标签
        return 返回每个参数的偏导
        """

        l0_in = img + self.parameters[0]['b']
        l0_out = self.tanh(l0_in)

        l1_in = np.dot(l0_out, self.parameters[1]['w']) + self.parameters[1]['b']
        l1_out = self.tanh(l1_in)

        l2_in = np.dot(l1_out, self.parameters[2]['w']) + self.parameters[2]['b']
        l2_out = self.softmax(l2_in)

        # y_pred - y
        diff = l2_out - self.onehot_encode(label)

        # 计算每个参数的梯度
        grad_b2 = 2*np.dot(diff, self.diffential[self.activetion[2]](l2_in))
        grad_w2 = np.outer(l1_out, grad_b2)
        # tanh的导数是一个以为矩阵，而不是对角矩阵，这里直接对应相乘
        grad_b1 = self.diffential[self.activetion[1]](l1_in)*np.dot(self.parameters[2]['w'], grad_b2)
        grad_w1 = img @ grad_b1

        grad_b0 = np.dot(self.parameters[1]['w'] , grad_b1)

        res = {'grad_w2':grad_w2,
               'grad_b2':grad_b2,
               'grad_w1':grad_w1,
               'grad_b1':grad_b1,
               'grad_b0':grad_b0}

        return res


    def predict_label(self, img):
        """输入一张图片根据参数预测结果
        :param img 模型输入，图片像素拉成的的一维的矩阵
        :param parameters 模型的参数
        """
        # 前向传播
        # 首先对输入加入偏置项
        l0_in = img + self.parameters[0]['b']
        l0_out = self.tanh(l0_in)

        l1_in = np.dot(l0_out, self.parameters[1]['w']) + self.parameters[1]['b']
        l1_out = self.tanh(l1_in)

        l2_in = np.dot(l1_out, self.parameters[2]['w']) + self.parameters[2]['b']
        l2_out = self.softmax(l2_in)

        predict = l2_out.argmax()
        return predict

