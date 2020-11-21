#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/20 下午3:27
# @Author  : lovemefan
# @File    : MYLSTM.py
import torch
from torch import nn

class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """单个LSTM节点
        :param input_size: 输入大小
        :param hidden_size: 隐藏层大小
        :param bias:
        :param num_chunks:
        """
        super(MyLSTMCell, self).__init__()
        num_chunks = 4
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias



    def forward(self, input, hx=None):
        """前向传播
        :param input 样本输入 大小为[batch_size, input_size]
        :param hx (c_t-1,ht-1)上一个LSTM的两个输出 大小为都为[batch_size, hidden_size]
        """
        if hx == None:
            zeros = torch.zeros(input.size(0), self.hidden_size)
            hx = (zeros, zeros)


class MyLSTM(nn.Module):
    """
    Args:
        input_size: 样本输入embedding的维度
        hidden_size: 隐藏层的大小
        num_layers: 循环神经网络的层数  例如``num_layers=2``
                    表示两个LSTM堆成的网络，第一个的LSTM的输出作为第二个LSTM的输入
        bias: 默认为False，表示不使用偏置向b_ih和b_hh
        bidirectional: 默认为False，是否为双向LSTM

    Input：input 为(h_0, c_0)
    输入的维度为：(seq_len, batch, input_size)  seq_len 为序列的长度，batch为batch的维度
    """

    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0.,  bidirectional=False):
        """
        :param input_size: 样本输入embedding的维度
        :param hidden_size: 隐藏层的大小
        :param num_layers: 循环神经网络的层数  例如``num_layers=2``表示两个LSTM堆成的网络，第一个的LSTM的输出作为第二个LSTM的输入
        :param bias: 默认为False，表示不使用偏置向b_ih和b_hh
        :param bidirectional: 默认为False，是否为双向LSTM
        """
        super(MyLSTM, self).__init__()

        gate_size = 4 * hidden_size
        num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(num_directions):


    def forward(self):
        pass
