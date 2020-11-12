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

        self.distribution = [
            # 第一层,初始化b的取值范围
            {'b': [0, 0]},
            {'b': [0, 0], 'w': [-math.sqrt(6 / (self.dimensions[0] + self.dimensions[1])),
                                math.sqrt(6 / (self.dimensions[0] + self.dimensions[1]))]},
            {'b': [0, 0], 'w': [-math.sqrt(6 / (self.dimensions[1] + self.dimensions[2])),
                                math.sqrt(6 / (self.dimensions[1] + self.dimensions[2]))]}
        ]

        if parameters:
            self.parameters = parameters
        else:
            self.parameters = self.init_parameters()

    def tanh(x):
        """tanh激活函数
        :return tanh(x) = (e^x - e^-x)/(e^x + e^-x)"""
        return np.tanh(x)

    def softmax(x):
        """
        softmax(x) = e^(x_i)/sum(e^(x_i))"""
        exp = np.exp(x - x.max())
        return exp / exp.sum()

    def sigmoid(x):
        """sigmoid 激活函数
        :return sigmoid(x) = 1/（1 + e^-x ）
        """
        return 1 / (1 + np.exp(-x))

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

    def predict_label(self, img, parameters):
        """输入一张图片根据参数预测结果
        :param img 图片像素拉成一维的矩阵
        :param parameters 模型的参数
        """

        # 首先对输入加入偏置项
        l0_in = img + parameters[0]['b']
        l0_out = self.tanh(l0_in)

        l1_in = np.dot(l0_out, parameters[1]['w']) + parameters[1]['b']
        l1_out = self.tanh(l1_in)

        l2_in = np.dot(l1_out, parameters[2]['w']) + parameters[2]['b']
        l2_out = self.softmax(l2_in)

        predict = l2_out.argmax()
        return predict
