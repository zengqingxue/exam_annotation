#! -*- coding: utf-8 -*-
import json
import random
import pandas as pd 
import numpy as np


def load_data(file_path):
    """加载数据
    单条格式：(文本, 标签id)
    """
    sample_list = []
    label_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t") # line = ['label1|label2','sample_text']
            label_list.append(line[1].split(" "))
            sample_list.append(line[2])

    text_len = [len(text) for text in sample_list]
    df = pd.DataFrame()
    df['len'] = text_len
    print('训练文本长度分度')
    print(df['len'].describe())

    return sample_list,label_list


if __name__ == '__main__':

    x,y= load_data('./data/multi-classification-train.txt')
    print(y[:5])
    

