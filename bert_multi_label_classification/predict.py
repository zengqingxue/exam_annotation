#! -*- coding: utf-8 -*-
import pickle
from tqdm import tqdm
from sklearn.metrics import hamming_loss, classification_report
from train import *

mlb = pickle.load(open('./checkpoint/mlb.pkl','rb'))
print(mlb.classes_.tolist())
threshold = 0.5



def predict_single(test_text):
	token_ids, segment_ids = tokenizer.encode(test_text, maxlen=maxlen)
	pred = model.predict([[token_ids], [segment_ids]])

	label_index = np.where(pred[0]>threshold)[0] # 取概率值大于阈值的 onehot 向量索引, [12,34]
	labels = [mlb.classes_.tolist()[i] for i in label_index]
	one_hot_label = np.where(pred[0]>threshold,1,0) # [[0,0,1,0,0,..],[0,0,1,0,1,..]]
	return one_hot_label, '|'.join(labels)


def evaluate():
    test_x,test_y = load_data('./data/multi-classification-test.txt')
    true_y_list = mlb.transform(test_y)

    pred_y_list = []
    for text in tqdm(test_x):
        pred_y, pred_label = predict_single(text)
        pred_y_list.append(pred_y)
    # F1值
    print(classification_report(true_y_list, pred_y_list, digits=4,target_names=mlb.classes_.tolist()))#


if __name__ == '__main__':
	test_text = '据36氪报道，近日，多名蔚来汽车离职员工透露，蔚来汽车正陆续裁员。'
	one_hot,label = predict_single(test_text)
	print("测试样本预测标签：",label)

	evaluate()