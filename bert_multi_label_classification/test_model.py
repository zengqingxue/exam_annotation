#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : test_model.py
#@ide    : PyCharm
#@time   : 2022-05-11 22:46:08
# '''
#
# import numpy as np
# import os
# from collections import Counter
# os.environ['TF_KERAS'] = '1'
# from bert4keras.backend import keras, K
# from bert4keras.models import build_transformer_model
# from bert4keras.tokenizers import Tokenizer
# from bert4keras.snippets import sequence_padding
# # from bert4keras.snippets import uniout
# from keras.models import Model
# # import pandas as pd
# import tensorflow as tf
# from config import Config
# config = Config()
#
#
# # bert配置
# config_path = config.config_path
# checkpoint_path = config.checkpoint_path
# dict_path = config.dict_path
#
# # 建立分词器
# tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
#
# # 建立加载模型
# bert = build_transformer_model(
#     config_path,
#     checkpoint_path,
#     model="albert",
#     with_pool='linear',
#     application='unilm',
#     return_keras_model=False,
# )
#
# encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
# export_path = "/data/zengqingxue/exam_annotation/bert_multi_label_classification/test"
# # base = 'test'
# model_version = "1"
# # encoder.save(base + r'\150k\1',save_format='tf') # <====注意model path里面的1是代表版本号，必须有这个不然tf serving 会报找不到可以serve的model
#
# encoder.save("test/test.h5")
# tf.keras.models.save_model(encoder, save_format="tf",filepath= export_path +"+/" +model_version)


import pandas as pd
df1 = pd.DataFrame({
    "id":[1,2,3,4],
    "label":["A B","B C","C D","E"],
    "title":["A1","B1","C1","D1"],
})

df2 = pd.DataFrame({
    "id":[1,2,3],
    "label":["A","B","C"],
    "title":["A1","B1","C1"],
})

import pandas as pd
d1 = pd.read_csv("../data/news/multi_cls/samples/samples_all.csv",sep="\t",names=['id','label',"content"])
d2 = pd.read_csv("../data/news/multi_cls/news_33.csv",sep="\t",names=['id','label',"content"])
print("df1, df2: ",df1.shape,df2.shape)
print("df1, df2: ",df1,df2)
intersected_df = pd.merge(df1, df2, on=['id'], how='inner')
intersected_df = intersected_df[['id','label_x','content_x']]
intersected_df.to_csv("./",sep="\t",header=None,index=None)
# intersected_df = pd.merge(df1, df2, how='inner')
print(intersected_df)
