#!/usr/bin/python3
# @Time    : 2020/9/25 上午11:23
# @Author  : lovemefan
# @File    : week02_2.py
# 编写程序，提取网页中的URL（超链接）。输入是html文件，输出为该页面中的URL
# 注：方法之一是(注意不唯一)扫描html文件中的："<a href=\"http://"串，该串后面的即URL。html文件自选，或用直接用sina.html

import re

import requests

# 从文件提取url
def find_all_links_from_file(file):
    # 去掉重复
    result = set()
    # 打开文件
    with open(file,'r',encoding='gbk') as file:
        html = file.read()
    links = re.findall('<a.*?href="(http.*?)"',html)

    for link in links:
        result.add(link)
    return result

#从url网页中提取url
def find_all_links_from_url(url):
    # 设置集合保证不重复
    result = set()
    html = requests.get(url).text
    links = re.findall('<a.*?href="(http.*?)"', html)
    # 加入集合，去掉重复
    for link in links:
        result.add(link)
    return result



if __name__ == '__main__':

    links1 = find_all_links_from_file('../data/sina.html.txt')
    links2 = find_all_links_from_url('https://www.sina.com.cn/')
    print(len(links1),len(links2))


