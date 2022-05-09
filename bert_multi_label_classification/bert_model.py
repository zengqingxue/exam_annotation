#! -*- coding: utf-8 -*-
from bert4keras.backend import keras,set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
import os
from config import Config
config = Config()
dict_path = config.dict_path
config_path = config.config_path
checkpoint_path = config.checkpoint_path
bert4keras_model_name = config.bert4keras_model_name
learning_rate = config.learning_rate


set_gelu('tanh')#relu

# def textcnn(inputs,kernel_initializer):
#     # 3,4,5
#     cnn1 = keras.layers.Conv1D(
#             256, #[[0.1,0.2],[0.3,0.1],[0.4,0.2]],[[0.12,0.32],[0.31,0.12],[0.24,0.12]]
#             3,
#             strides=1,
#             padding='same',#'valid'
#             activation='relu',
#             kernel_initializer=kernel_initializer
#         )(inputs) # shape=[batch_size,maxlen-2,256]
#     cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]
#
#     cnn2 = keras.layers.Conv1D(
#             256,
#             4,
#             strides=1,
#             padding='same',
#             activation='relu',
#             kernel_initializer=kernel_initializer
#         )(inputs)
#     cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)
#
#     cnn3 = keras.layers.Conv1D(
#             256,
#             5,
#             strides=1,
#             padding='same',
#             kernel_initializer=kernel_initializer
#         )(inputs)
#     cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)
#
#     output = keras.layers.concatenate(
#         [cnn1,cnn2,cnn3],
#         axis=-1) #[batch_size,256*3]
#
#     return output

def build_bert_model(config_path,checkpoint_path,class_nums):
    bert = build_transformer_model(
        config_path=config_path, 
        checkpoint_path=checkpoint_path, 
        model=bert4keras_model_name,
        return_keras_model=False)

    cls_features = keras.layers.Lambda(
        lambda x:x[:,0],
        name='cls-token'
        )(bert.model.output) #shape=[batch_size,768]
    # all_token_embedding = keras.layers.Lambda(
    #     lambda x:x[:,1:-1],
    #     name='all-token'
    #     )(bert.model.output) #shape=[batch_size,maxlen-2,768]
    #
    # cnn_features = textcnn(
    #   all_token_embedding,'he_normal') #shape=[batch_size,cnn_output_dim]
    # concat_features = keras.layers.concatenate(
    #   [cls_features,cnn_features],
    #   axis=-1)

    # concat_features = keras.layers.Dropout(0.2)(concat_features)
    # dense = keras.layers.Dense(
    #       units=256,
    #       activation='relu',
    #       kernel_initializer='he_normal'
    #   )(concat_features)

    # output = keras.layers.Dense(
    #         units=class_nums,
    #         activation='sigmoid', # 多分类模型变多标签模型 softmax --> sigmoid
    #         kernel_initializer='he_normal'
    #     )(dense)

    output = keras.layers.Dense(
        units=class_nums,
        activation='sigmoid',  # 多分类模型变多标签模型 softmax --> sigmoid
        kernel_initializer='he_normal'
    )(cls_features)

    model = keras.models.Model(bert.model.input,output)
    model.compile(
        loss='binary_crossentropy',#二分类交叉熵损失函数
        # optimizer=Adam(5e-5),
        optimizer=Adam(learning_rate),
        metrics=['accuracy'],
    )

    return model

if __name__ == '__main__':
    # config_path='E:/bert_weight_files/bert_wwm/bert_config.json'
    # checkpoint_path='E:/bert_weight_files/bert_wwm/bert_model.ckpt'
    config_path = config_path
    checkpoint_path = checkpoint_path
    class_nums=30
    build_bert_model(config_path, checkpoint_path, class_nums)