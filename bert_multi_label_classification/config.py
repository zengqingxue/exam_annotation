#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : config.py
#@ide    : PyCharm
#@time   : 2022-05-09 10:49:13
'''
import os,sys
cur_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cur_dir,os.path.pardir))
model_dir_name = "albert_tiny_google_zh_489k"   # "chinese_L-12_H-768_A-12"  "albert_tiny_google_zh_489k"
config_path= os.path.abspath(os.path.join(parent_dir,"bert_multi_label/pretrained_model/{}/bert_config.json".format(model_dir_name)))
checkpoint_path = os.path.abspath(os.path.join(parent_dir,"bert_multi_label/pretrained_model/{}/bert_model.ckpt".format(model_dir_name)))
dict_path = os.path.abspath(os.path.join(parent_dir,"bert_multi_label/pretrained_model/{}/vocab.txt".format(model_dir_name)))


class Config():
    def __init__(self):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path
        self.mlb_path =  './checkpoint/mlb.pkl'
        self.best_model_filepath =  './checkpoint/best_model.weights'
        self.train_data =  './data/multi-classification-train.csv'
        self.test_data =  './data/multi-classification-test.csv'

        self.prob_threshold = 0.5
        self.bert4keras_model_name = "albert"     # "bert" "albert"
        self.learning_rate = 1e-4     # "bert" "albert"   1e-4  5e-5
        self.epochs = 3
        self.class_nums = 30
        self.maxlen = 256
        self.batch_size = 64

if __name__ == '__main__':
    print("config_path: ",config_path)
    print("checkpoint_path: ",checkpoint_path)
    print("dict_path: ",dict_path)
    pass



