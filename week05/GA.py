#!/usr/bin/python3
# @Time    : 2020/10/26 下午4:08
# @Author  : lovemefan
# @File    : GA.py
import random
from time import sleep

import numpy as np
import matplotlib.pyplot as plt


class GA:
    """
    遗传算法
    1. 初始化种群
    2. 交叉
    3. 变异
    4. 合并
    5. 选择
    """
    def __init__(self):
        # 交叉模式
        self.SINGLE_POINT = 0
        self.DOUBLE_POINT = 1


    def species_initial(self, population_size, chromosome_length):
        """初始化种群,每一个个体就是一个解
        :param population_size 种群大小
        :param chromosome_length 染色体长度，二进制表示
        """
        self.population_size = population_size

        # 种群一个而为矩阵表示，采用二进制编码
        self.population = np.around(np.random.rand(population_size, chromosome_length)).astype(np.int16)
        print(f"初始化种群，种群大小为{population_size},染色体用{chromosome_length}bit表示")
        return self.population

    def decode(self, chromosome, spoint):
        """染色体解码，二进制转化成十进制,采用小端模式
        第一位当作符号位
        :param chromosome 染色体数组，
        :param sponit 二进制中整数的位数
        """
        d = []
        for item in chromosome:

            num = 0
            for i in range(len(item[1:])):
                num += item[i] * np.power(2,i)
            # 符号位
            sign = -1 if item[0] == 1 else 1
            d.append((sign*num) / 2**(len(item) - 1 - spoint))
        return d

    def crossover(self, point, mode=0):
        """交叉
        :param point 表示从第几个点之后的就进行交换
        :param mode 0 表示单点交换 1 表示双点交叉
        :return offspring 返回交叉后的后代
        """
        offspring = []

        if mode == self.SINGLE_POINT:
            print("使用单点交叉模式")
            if point < 0 or point >= self.population.shape[0]:
                raise Exception(f"当前种群大小为：{self.population.size[0]}；point：{point} 越界")
            else:
                # 每两个交换产生新的个体
                len = self.population.shape[0] - 1
                print(f"染色体交叉产生子代中")
                for i in range(len):
                    offspring.append(np.append(self.population[i][:point], self.population[i+1][point:]).tolist())
                print(f"染色体交叉产生子代完成，生成了{offspring.__len__()}个新个体")

        if mode == self.DOUBLE_POINT:
            # TODO 双点交叉
            pass
        #
        return offspring

    def mutation(self, offspring, n):
        """变异，随机取n个取反
        :param offspring 传入待变异的子代
        :param n n表示随机选二进制的n位进行反转
        :return offspring 返回变异后的子代
        """
        print(f"{offspring.__len__()}个新个体变异中")
        for i in range(len(offspring)):
            chromosome = offspring[i]
            temp = [0 for _ in range(len(chromosome) - n)]
            temp += [1 for _ in range(n)]
            # 符号也可能突变
            random.shuffle(temp)
            reverse = lambda x: 1 if x == 0 else 0
            chromosome = [chromosome[i] if temp[i]==0 else reverse(chromosome[i]) for i in range(len(chromosome)) ]
            offspring[i] = chromosome
        print("变异完成")
        return offspring

    def merge(self, offspring):
        """将父代与子代合并"""
        print("合并父代与子代")
        self.population = np.vstack((self.population, offspring))

    def fitness(self, x):
        """计算适应度，适应度不能为负值，所以要加一个比较大的正值"""
        return self.function(x) + 100

    def selection(self):
        """轮盘赌选择个体"""
        # 计算适应度
        print("选择过程中")
        fintnesses = self.fitness(self.decode(self.population, 3))
        # 计算被抽中的概率
        p = np.divide(fintnesses, np.sum(fintnesses)).tolist()
        # 轮盘赌选择
        choise = np.random.choice(self.population.shape[0], self.population_size, replace=False, p=p)

        self.population = self.population[choise]
        return self.population

    def function(self, x):
        x = np.array(x)
        return -np.power(x, 2) + 10*np.cos(2*np.pi*x + 30)


if __name__ == '__main__':
    population_size = 200
    chromosome_length = 10


    ga = GA()
    ga.species_initial(population_size, chromosome_length)
    # 设置为尽可能小
    max_x = -0xfffff;
    max_y = -0xfffff
    plt.ion()

    for i in range(100):
        offspring = ga.crossover(4)
        offspring = ga.mutation(offspring, 1)
        ga.merge(offspring)
        ga.selection()
        max_y = max(ga.function(ga.decode(ga.population, 3)))
        print(max_y)

        plt.clf()
        # 画函数
        x = np.linspace(-8, 8, 100)
        y = ga.function(x)
        plt.plot(x, y)

        # 画 种群
        x = ga.decode(ga.population, 3)
        y = ga.function(x)

        colors1 = '#DC143C'
        area = np.pi * 4   # 点面积
        plt.scatter(x, y, s=area, c=colors1, alpha=0.4, label='类别A')

        # plt.draw()
        plt.pause(0.1)

    input('press any key to exit')
