#! -*- coding: utf-8 -*-
import json
import pickle
import pandas as pd 
import numpy as np 
import random
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam

from bert_model import build_bert_model
from data_helper import load_data

#定义超参数和配置文件
epochs = 30
class_nums = 65
maxlen = 200
batch_size = 16

config_path='E:/bert_weight_files/roberta/bert_config_rbt3.json'
checkpoint_path='E:/bert_weight_files/roberta/bert_model.ckpt'
dict_path = 'E:/bert_weight_files/roberta/vocab.txt'

bast_model_filepath = './checkpoint/best_model.weights'

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
        pred = np.where(pred[0]>0.5,1,0)
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
        # model.load_weights(bast_model_filepath)
        acc = cal_acc(test_x,test_y) # 计算多标签分类准确率
        if acc > self.best_acc:
            print('Accuracy increased from {} to {} ,save model to {}'.format(self.best_acc,acc,bast_model_filepath))
            self.best_acc = acc 
            self.wait = 0
            model.save_weights(bast_model_filepath)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        print('valid best acc: %.5f\n' % (self.best_acc))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


model = build_bert_model(config_path,checkpoint_path,class_nums)


if __name__ == '__main__':
    # 加载数据集
    train_x,train_y = load_data('./data/multi-classification-train.txt')
    test_x,test_y = load_data('./data/multi-classification-test.txt')

    shuffle_index =[i for i in range(len(train_x))]
    random.shuffle(shuffle_index)
    train_x = [train_x[i] for i in shuffle_index]
    train_y = [train_y[i] for i in shuffle_index]
    
    mlb = MultiLabelBinarizer()
    mlb.fit(train_y)
    print("标签数量：",len(mlb.classes_))
    class_nums = len(mlb.classes_)
    pickle.dump(mlb, open('./checkpoint/mlb.pkl','wb'))

    train_y = mlb.transform(train_y) # [[label1,label2],[label3]] --> [[1,1,0],[0,0,1]]
    test_y = mlb.transform(test_y)

    train_data = [[x,y.tolist()] for x,y in zip(train_x,train_y)] # 将相应的样本和标签组成一个tuple
    print(train_data[:3])
    test_data = [[x,y.tolist()] for x,y in zip(test_x,test_y)] # --> [[x1,y1],[x2,y2],[],..]

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    print(model.summary())

    evalutor = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        # validation_data=test_generator.forfit(), 
        # validation_steps=len(test_generator),
        shuffle=True, 
        callbacks=[evalutor]
    )
else:
    model.load_weights(bast_model_filepath)
