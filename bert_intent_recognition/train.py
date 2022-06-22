#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : train.py.py
#@ide    : PyCharm
#@time   : 2022-05-04 22:40:52
'''
#! -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam

from bert_model import build_bert_model
from data_helper import load_data
from config import Config
config = Config()

# region 定义超参数和配置文件
class_nums = config.class_nums
maxlen = config.maxlen
batch_size = config.batch_size

config_path = config.config_path
checkpoint_path = config.checkpoint_path
dict_path = config.dict_path
best_model_filepath = config.best_model_filepath
# endregion

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
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

if __name__ == '__main__':
    # 加载数据集
    train_data = load_data(config.train_data_file)
    test_data = load_data(config.test_data_file)

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    model = build_bert_model(config_path,checkpoint_path,class_nums)
    # model.load_weights("./checkpoint/best_model.weights")
    model.load_weights(best_model_filepath)

    print(model.summary())
    model.compile(
        loss='sparse_categorical_crossentropy',  # 单标签多分类损失函数
        # loss='binary_crossentropy',  # 多标签分类的损失函数，二分类交叉熵损失函数
        optimizer=Adam(config.learning_rate),
        metrics=['accuracy'],
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=1,
        verbose=2,
        mode='min'
        )

    checkpoint = keras.callbacks.ModelCheckpoint(
        best_model_filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
        )

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=config.epochs,
        validation_data=test_generator.forfit(),
        validation_steps=len(test_generator),
        shuffle=True,
        callbacks=[earlystop,checkpoint]
    )

    model.load_weights(best_model_filepath)
    test_pred = []
    test_true = []
    for x,y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:,1].tolist()
    print(set(test_true))
    print(set(test_pred))

    target_names = [line.strip() for line in open('label','r',encoding='utf8')]
    print(classification_report(test_true, test_pred,target_names=target_names))
