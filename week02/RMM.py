#!/usr/bin/python3
# @Time    : 2020/9/26 下午8:12
# @Author  : lovemefan
# @File    : RMM.py
# 逆向最大匹配算法（Reverse Maximum Match Method，RMM）
class RMM(object):
    def __init__(self,dic_path,stop_word_path):
        # 存放字典
        self.dic = set()
        # 词最大长度
        self.maximum = 0
        # 读取文件，保存成字典
        with open(dic_path,'r',encoding='utf-8') as f:
            for line in f:
                # 先清洗数据后分割
                line = line.strip()
                line = line.split(' ')[0]
                self.dic.add(line)
                # 存储字典中单词长度最大值
                self.maximum = max(self.maximum,len(line))
    # 分词
    def cut(self, text):
        #存放分词结果
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximum, 0, -1):
                if index - size < 0:
                    continue
                piece = text[(index - size):index]
                if piece in self.dic:
                    word = piece
                    result.append(word)
                    index -= size
                    break
            if word is None:
                index -= 1
        return result[::-1]
    def cut_for_search(self):
        pass
    def is_stop_word(self,word):
        return word in self.stop_word
