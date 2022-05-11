#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : test_model.py
#@ide    : PyCharm
#@time   : 2022-05-11 22:46:08
'''

import numpy as np
import os
from collections import Counter
os.environ['TF_KERAS'] = '1'
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
# from bert4keras.snippets import uniout
from keras.models import Model
# import pandas as pd
from config import Config
config = Config()


# bert配置
config_path = config.config_path
checkpoint_path = config.checkpoint_path
dict_path = config.dict_path

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    model="albert",
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
export_path = "abc"
base = 'test'

encoder.save(base + r'\150k\1',save_format='tf') # <====注意model path里面的1是代表版本号，必须有这个不然tf serving 会报找不到可以serve的model
#
