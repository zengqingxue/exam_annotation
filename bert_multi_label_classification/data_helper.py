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
            # line = line.strip().split(" ") # line = ['label1|label2','sample_text']
            # label_list.append(line[0].split("|"))
            # sample_list.append(line[1])

    text_len = [len(text) for text in sample_list]
    df = pd.DataFrame()
    df['text_len'] = text_len
    df['label'] = label_list
    print('{}文本长度分度分布'.format(file_path))
    print(df['text_len'].describe(percentiles=[0.25,0.5,0.75,0.8,0.9,0.95]))
    print('{}文本标签分布'.format(file_path))
    point_df = df.sample(frac=1.0)
    print(point_df['label'].value_counts())
    print("sample_list[:2]: {}",sample_list[:2])
    print("label_list[:2]: {}",label_list[:2])
    import matplotlib.pyplot as plt
    plt.hist(point_df['text_len'], bins=30, rwidth=0.9, density=True, )
    plt.hist(point_df['text_len'], bins=30, rwidth=0.9 )
    plt.show()

    return sample_list,label_list


if __name__ == '__main__':

    x,y= load_data('./data/multi-classification-train.txt')
    print(y[:5])
    

