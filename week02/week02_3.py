#!/usr/bin/python3
# @Time    : 2020/9/25 下午3:37
# @Author  : lovemefan
# @File    : test.py

# import jieba
#
word = "昆明理工大学信息工程与自动化学院计算机系开设课程如下"
# words = jieba.cut(word)
# print(','.join(words))
#
# words = jieba.cut_for_search(word)
# print(','.join(words))
from week02.RMM import RMM
dic_path = '../data/dict.txt'
stop_word_path = '../data/stop_word.txt'
r = RMM(dic_path, stop_word_path)
text = r.cut(word)
print(r.maximum)
print(text)