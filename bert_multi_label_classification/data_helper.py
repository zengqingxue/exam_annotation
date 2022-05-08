#! -*- coding: utf-8 -*-
import json
import random
import pandas as pd 
import numpy as np
import argparse
from loguru import logger
logger.add('./logs/my.log', format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - {module} - {function} - {level} - line:{line} - {message}", level="INFO",rotation='00:00',retention="3 day")
from sklearn.model_selection import train_test_split

def analyze_textlen_labels(df):
    logger.info('{}文本标签数据分布{}',"===="*4,"===="*4)
    df = df.sample(frac=1.0)
    logger.info(df['label'].value_counts())

    logger.info('{}文本长度分度分布{}', "====" * 10, "====" * 10)
    df['text_len'] = df['content'].map(lambda x: len(x))
    logger.info(df['text_len'].describe(percentiles=[0.25, 0.5, 0.75, 0.8, 0.9, 0.95]))
    # import matplotlib.pyplot as plt
    # plt.hist(df['text_len'], bins=30, rwidth=0.9, density=True, )
    # plt.hist(df['text_len'], bins=30, rwidth=0.9,)
    # plt.show()


def split_train_test(df,test_size=0.2,is_names=False,names=None,is_valid=False):
    if is_names:
        df = df[names]
    df.dropna(inplace=True)
    logger.info("the dataset : {}\n", df)
    logger.info("The shape of the dataset : {}", df.shape)
    logger.info("开始划分数据集 ...... ")
    df_train, df_test = train_test_split(df[:], test_size=test_size, shuffle=True)
    if is_valid:
        df_valid, df_test = train_test_split(df_test[:], test_size=0.5, shuffle=True)
    return df_train,df_test


def split_train_test(file_path):
    """切分训练集和测试集，并进行数据分析"""
    # point_df = pd.read_table(file_path, sep=" ", header=None, names=["label", "content"])
    point_df = pd.read_table(file_path, sep="\t", header=None, names=["id", "label", "content"])
    point_df = point_df[["label", "content"]]
    point_df.dropna(inplace=True)
    logger.info("\nthe dataset : {}\n", point_df)
    logger.info("\nThe shape of the dataset : {}\n", point_df.shape)

    # 数据分析
    analyze_textlen_labels(point_df)

    logger.info("\n划分数据集 ... \n")
    df_train, df_test = train_test_split(point_df[:], test_size=0.2, shuffle=True)
    # df_valid, df_test = train_test_split(df_test[:], test_size=0.5, shuffle=True)
    print("df_train",df_train)
    df_train.to_csv("./data/multi-classification-train1.csv",sep="\t",header=None,index=None)
    df_test.to_csv("./data/multi-classification-test1.csv",sep="\t",header=None,index=None)


def load_data(file_path):
    """加载数据
    单条格式：(文本, 标签id)
    """
    sample_list = []
    label_list = []
    # label_list1 = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t") # line = ['label1 label2','sample_text']
            label_list.append(line[0].split(" "))
            # label_list1.append(line[1])
            sample_list.append(line[1])
            # line = line.strip().split(" ") # line = ['label1|label2','sample_text']
            # label_list.append(line[0].split("|"))
            # sample_list.append(line[1])

    # 分文本长度及标签分布
    # label_list.append(line[1].split(" "))
    # text_len = [len(text) for text in sample_list]
    # df = pd.DataFrame()
    # df['text_len'] = text_len
    # df['label'] = label_list1
    # analyze_textlen_labels(df)

    logger.info("sample_list[:2]: {}",sample_list[:2])
    logger.info("label_list[:2]: {}",label_list[:2])

    return sample_list,label_list

def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument("--oprate",dest="oprate",type=str,help="please input oprate type \n e.g: python data_helper.py --oprate split")
    args = parse.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arg()
    oprate_option = args['oprate']
    file_path = "./data/news_33.csv.1"
    if oprate_option == "split":
        split_train_test(file_path)
    elif oprate_option == "load":
        load_data(file_path)
        x, y = load_data('./data/multi-classification-train.txt')
        logger.info(y[:5])
    else:
        pass



    
    

