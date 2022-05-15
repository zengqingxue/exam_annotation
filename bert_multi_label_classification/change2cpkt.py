#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : change2cpkt.py
#@ide    : PyCharm
#@time   : 2022-05-10 09:43:46
'''
#! -*- coding: utf-8 -*-
import json
import pickle
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import load_model
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam
import os,sys,time
from bert_model import build_bert_model
from data_helper import load_data
from loguru import logger
logger.add('./logs/my.log', format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - {module} - {function} - {level} - line:{line} - {message}", level="INFO",rotation='00:00',retention="3 day")

from config import Config
config = Config()
dict_path = config.dict_path
config_path = config.config_path
checkpoint_path = config.checkpoint_path
bert4keras_model_name = config.bert4keras_model_name
epochs = config.epochs
class_nums = config.class_nums
maxlen = config.maxlen
batch_size = config.maxlen
train_data = config.train_data
test_data = config.test_data
best_model_filepath = config.best_model_filepath
mlb_path = config.mlb_path
prob_threshold = config.prob_threshold

import argparse
#
# def parse_arg():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs",dest=epochs,type=int,default=3,help="the number of train epochs")
#     parser.add_argument("--class_nums",dest=class_nums,type=int,default=30,help="the number of samples class_nums")
#     parser.add_argument("--maxlen",dest=maxlen,type=int,default=200,help="the maxlen of text ")
#     parser.add_argument("--batch_size",dest=batch_size,type=int,default=16,help="the batch_size of train ")
#     parser.add_argument("--model_path",dest=model_path,type=str,default="",help="the model version of train samples number")
#     parser.add_argument("--data_version",dest=data_version,type=str,default="",help="the model version of train samples number")
#     args = parser.parse_args()
#     return vars(args)



#定义超参数和配置文件
# args = parse_arg()
# epochs = args['epochs']
# class_nums = args['class_nums']
# maxlen = args['maxlen']
# batch_size = args['batch_size']
# model_path = args['model_path']
# data_version = args['data_version']

if maxlen > 512:
    assert 0 is -1
    logger.info("train exits!!! maxlen of the input is larger than 512 which is the maxlen of pretrain bert")
# config_path='E:/bert_weight_files/roberta/bert_config_rbt3.json'
# checkpoint_path='E:/bert_weight_files/roberta/bert_model.ckpt'
# dict_path = 'E:/bert_weight_files/roberta/vocab.txt'

tokenizer = Tokenizer(dict_path)

class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)#[1,3,2,5,9,12,243,0,0,0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], np.asarray(batch_labels)
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def cal_acc(text_list,label_label):
    cnt = 1e-8
    total = len(text_list)
    for text,label in tqdm(zip(text_list,label_label)):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        pred = model.predict([[token_ids], [segment_ids]])
        pred = np.where(pred[0]>prob_threshold,1,0)
        cnt += 1-(label!=pred).any()

    return cnt/total

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.patience = 3
        self.best_acc = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self,epoch,logs=None):
        # model.load_weights(best_model_filepath)
        acc = cal_acc(test_x,test_y) # 计算多标签分类准确率
        if acc > self.best_acc:
            logger.info('Accuracy increased from {} to {} ,save model to {}'.format(self.best_acc,acc,best_model_filepath))
            self.best_acc = acc
            self.wait = 0
            model.save_weights(best_model_filepath)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        logger.info('valid best acc: %.5f\n' % (self.best_acc))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logger.info('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

model = build_bert_model(config_path,checkpoint_path,class_nums)
model.save_weights()
model.load_weights()
model.save()
load_model()