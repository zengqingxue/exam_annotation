#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : config.py
#@ide    : PyCharm
#@time   : 2022-05-06 17:27:37
'''
import os,sys
cur_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cur_dir,os.pardir))
pretrain_dir = os.path.join(parent_dir,"bert_multi_label")
pretrain_model_name = "chinese_L-12_H-768_A-12"

class Config(object):
    def __init__(self):
        self.cur_dir = cur_dir
        self.data_dir = os.path.join(cur_dir,"data")
        self.train_data_file = os.path.join(self.data_dir,"train.csv")
        self.test_data_file = os.path.join(self.data_dir,"test.csv")
        self.config_path = os.path.join(pretrain_dir,"pretrained_model/{}/bert_config.json".format(pretrain_model_name))
        self.checkpoint_path = os.path.join(pretrain_dir,"pretrained_model/{}/bert_model.ckpt".format(pretrain_model_name))
        self.dict_path = os.path.join(pretrain_dir,"pretrained_model/{}/vocab.txt".format(pretrain_model_name))
        self.label_file_name = "label"

        self.class_nums = len([line.strip() for line in open(self.label_file_name, 'r', encoding='utf8')])
        self.maxlen = 60
        self.batch_size = 16
        self.epochs = 3
        self.learning_rate = 1e-4
        self.best_model_filepath = './checkpoint/best_model.weights'


if __name__ == '__main__':
    config = Config()
    print(config.config_path)
    print(config.class_nums)
    print(pretrain_dir)
    pass