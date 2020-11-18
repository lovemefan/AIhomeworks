#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 下午9:29
# @Author  : lovemefan
# @File    : dataLoader.py
import os
import gzip
import requests
import struct
import numpy as np
from matplotlib import pyplot as plot


class DataLoader:
    def __init__(self):
        self.download_dataset()
        relative_path_dir = 'data/mnist'
        absolute_path_dir = os.path.join(os.path.dirname(os.getcwd()) , relative_path_dir)
        self.train_img_path = os.path.join(absolute_path_dir, 'train-images-idx3-ubyte')
        self.train_label_path = os.path.join(absolute_path_dir, 'train-labels-idx1-ubyte')
        self.test_img_path = os.path.join(absolute_path_dir, 't10k-images-idx3-ubyte')
        self.test_label_path = os.path.join(absolute_path_dir, 't10k-labels-idx1-ubyte')
        self.load_data()

    def download_dataset(self):
        """下载数据集
        如果发现文件损坏，删掉文件重新下载
        数据集来源网站 http://yann.lecun.com/exdb/mnist/
        一共有三个文件，其中包括60000个训练集，10000个测试集
        train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
        train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
        t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
        t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
        """
        # 以下是数据集各个文件的下载地址
        train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

        self.download_file(train_images_url, 'train-images-idx3-ubyte.gz')
        self.download_file(train_labels_url, 'train-labels-idx1-ubyte.gz')
        self.download_file(test_images_url, 't10k-images-idx3-ubyte.gz')
        self.download_file(test_labels_url, 't10k-labels-idx1-ubyte.gz')

    def download_file(self, url, file_name):
        """从网络上下载文件，如果文件已存在则跳过下载
        :param url 文件网络路径
        :param file_path 保存文件路径
        """
        # 取出文件夹的路径
        relative_path_dir = 'data/mnist'

        absolute_path_dir = os.path.join(os.path.dirname(os.getcwd()) , relative_path_dir)

        # 如果不存在就创建文件夹
        if not os.path.exists(absolute_path_dir):
            os.mkdir(absolute_path_dir)


        # 如果文件不存在
        absolute_path_file = os.path.join(absolute_path_dir, file_name)
        if not os.path.exists(absolute_path_file):
            print(f"正在下载{file_name}")
            response = requests.get(url)
            with open(absolute_path_file, 'wb') as f:
                f.write(response.content)

            print(f"{file_name}下载完毕")
            print(f"路径为：{absolute_path_file}")
            # 将文件夹解压开
            print(f"{file_name}正在解压")
            self.un_gz(absolute_path_file)
            print(f"{file_name}解压完成")
            print('\n')

    def un_gz(self, file_name):
        """ungz zip file"""

        f_name = file_name.replace(".gz", "")

        if not os.path.exists(f_name):
            # 获取文件的名称，去掉
            # 创建gzip对象
            g_file = gzip.GzipFile(file_name)
            # gzip对象用read()打开后，写入open()建立的文件里。
            open(f_name, "wb+").write(g_file.read())
            # 关闭gzip对象
            g_file.close()

    def load_data(self):
        """从文件中加载数据集"""
        # 加载训练集数据
        print('\033[34m')
        print("正在加载数据中")
        with open(self.train_img_path, 'rb') as f:
            magic_number, image_num, rows, columns = struct.unpack('>4i', f.read(16))
            print(f"训练集共有{image_num}张图片，大小为{rows}*{columns}")
            # 读取数据
            self.train_img = np.array(struct.unpack(f'>{image_num * rows * columns}B', f.read()),
                                      dtype=np.uint8).reshape(-1, rows * columns)

        # 加载训练集标签
        with open(self.train_label_path, 'rb') as f:
            magic_number, image_num = struct.unpack('>2i', f.read(8))
            self.train_label = np.array(struct.unpack(f'>{image_num}B', f.read()),
                                        dtype=np.uint8)

        # 加载测试集数据
        with open(self.test_img_path, 'rb') as f:
            magic_number, image_num, rows, columns = struct.unpack('>4i', f.read(16))
            print(f"测试集共有{image_num}张图片，大小为{rows}*{columns}")
            self.test_img = np.array(struct.unpack(f'>{image_num * rows * columns}B', f.read()),
                                     dtype=np.uint8).reshape(-1, rows * columns)

        # 加载测试集标签
        with open(self.test_label_path, 'rb') as f:
            magic_number, image_num = struct.unpack('>2i', f.read(8))
            self.test_label = np.array(struct.unpack(f'>{image_num}B', f.read()),
                                       dtype=np.uint8)

        print("加载数据完成")
        print('\033[0m')

    def get_train_batch(self, batch_size=None):
        """返回训练集的一个batch
        :returns (data, label) 返回一个元组，分别是数据矩阵与标签矩阵
        """
        # 如果没有指定batch_size 就返回所有数据
        if not batch_size: return (self.train_img, )

        data_shape = self.train_img.shape
        batch = data_shape[0] // batch_size

        for i in range(batch+1):
            yield (self.train_img[i*batch_size: min(data_shape[0], (i+1)*batch_size)],
                   self.train_label[i*batch_size: min(data_shape[0], (i+1)*batch_size)])

    def get_test_data_batch(self, batch_size=None):
        """返回测试集
        """
        if not batch_size:
            return (self.test_img, self.test_label)

    def console_out(self, text, color='b'):
        """向控制台输出有颜色的文字
        :param text 输出的文字
        :param color 输出的颜色，r 红色，b蓝色，g 绿色，y黄色
        """
        if color == 'r':
            print(f"\033[31m{text}\033[0m")
        elif color == 'b':
            print(f"\033[34m{text}\033[0m")
        elif color == 'g':
            print(f"\033[32m{text}\033[0m")
        elif color == 'y':
            print(f"\033[33m{text}\033[0m")
        else:
            print(f"\033[34m{text}\033[0m")


if __name__ == '__main__':
    # 下载数据集
    dataLoader = DataLoader()
    image = dataLoader.test_img[0].reshape(28,28)
    plot.title(f"{dataLoader.test_label[0]}")
    plot.imshow(image, cmap='gray')




