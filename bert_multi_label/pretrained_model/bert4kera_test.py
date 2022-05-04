#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : bert4kera_test.py
#@ide    : PyCharm
#@time   : 2022-05-04 15:57:53
'''
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
import sys,os
from loguru import logger
base_dir = os.path.dirname(__file__)

config_path = '{}/chinese_L-12_H-768_A-12/bert_config.json'.format(base_dir)
checkpoint_path = '{}/chinese_L-12_H-768_A-12/bert_model.ckpt'.format(base_dir)
dict_path = '{}/chinese_L-12_H-768_A-12/vocab.txt'.format(base_dir)

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
logger.info("token_ids: {} \n ,  segment_ids: {} ",token_ids, segment_ids)
predict_word = model.predict([np.array([token_ids]), np.array([segment_ids])])
logger.info('===== predicting =====')
logger.info("predict_word的形状： {}",predict_word.shape)