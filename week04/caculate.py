#!/usr/bin/python3
# @Time    : 2020/10/29 下午9:45
# @Author  : lovemefan
# @File    : caculate.py
import numpy as np

train_size = 14
outlook = {
        'sunny': (2, 3),
        'overcast': (4, 0),
        'rain': (3, 2)
}

temperature = {
    'hot': (2, 2),
    'mild': (4, 2),
    'coll': (3, 1)
}

humidlity = {
    'high': (3, 4),
    'normal': (6, 1)
}

wind = {
    'weak': (6, 2),
    'strong': (3, 3),
}

E = {}
S = {
    'S': {'S': (9, 5)},
    'outlook': outlook,
    'temperature': temperature,
    'humidlity': humidlity,
    'wind': wind,

}
Gain = {}
def entropy(a, b):
    """计算熵"""
    if a != 0 and b !=0 :
        return  -np.log2(a) * a - np.log2(b) * b
    elif a != 0 and b == 0:
        return -np.log2(a) * a
    elif a == 0 and b != 0:
        return -np.log2(b) * b
    else:
        return 0

def gain(s1,s2):
    """计算收益值"""
    s2_sum = 0
    for key, value in S[s2].items():
        s2_sum += -(value[0] + value[1])*E[key] / train_size
    Gain[f'{s1},{s2}'] = E[s1] + s2_sum
    return E[s1] + s2_sum



if __name__ == '__main__':

    for key, value in S.items():
        for item_key, item_value in value.items():
            # 计算每种情况的熵
            sum = item_value[0] + item_value[1]
            E[item_key] = entropy(item_value[0]/sum, item_value[1]/sum)

    for key,value in E.items():
        # 跳过S
        if key == 'S':
            print(f"E({key})= {value}")
        else:
            print(f"E(S_{key})= {value}")


    max_gain, max_gain_key = 0, ""
    for key,value in S.items():

        # 跳过S
        if key == 'S': continue
        max_gain, max_gain_key = (gain('S',key), key)  if gain('S',key) > max_gain else (max_gain, max_gain_key)

        gain('S',key)
        print(f'Gain(S,{key})={Gain[f"S,{key}"]}')

    print(f"max is Gain(S,{max_gain_key})={max_gain}, so we choose {max_gain_key}")

