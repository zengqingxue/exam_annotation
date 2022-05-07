#coding:utf-8
import numpy as np
# forked from DengYangyong/exam_annotation  https://github.com/DengYangyong/exam_annotation
import os,pathlib
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

class Config(object):
    def __init__(self):
        self.point_path = os.path.join(root,"data","news/multi_cls/news_33.csv.1")
        self.proc_dir = os.path.join(root,"data","bert_multi_label_results","proc")
        self.class_path = os.path.join(root,"data","news/multi_cls","33_class.txt")
        self.output_dir = os.path.join(root,"data","bert_multi_label_results")
        self.vocab_file = os.path.join("pretrained_model","chinese_L-12_H-768_A-12","vocab.txt")
        self.prob_threshold= 0.5

        # self.max_len = 512
        # self.output_dim = 95
        # self.learning_rate = 2e-5
        # self.num_epochs = 3

if __name__ == '__main__':
    print("root")