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
	return one_hot_label, ' '.join(labels)


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
	test_text = '南京疫情外溢6省11市感染至少170人病例情况一图读懂__自7月20日南京禄口机场检出9例阳性病例以来截至7月27日24时南京本轮疫情累计报告155例新冠感染者根据已公布的病例详情有90余人都是该机场的工作人员66人系从事保洁工作另外还涉及到机场的司机地勤辅警水电工等加上广东中山辽宁沈阳安徽和县江苏宿迁四川绵阳安徽芜湖广东珠海四川泸州四川成都辽宁大连湖南常德6省11市报告的相关联感染者南京此轮疫情感染者总计至少达170人'
	# test_text = '陕西疾控这些人员属于高风险人群请立即报告__西部网讯记者马晴茹因大连成都常德等地连日来报告的多例确诊病例和无症状感染者均在湖南省张家界等地有时间和空间上的轨迹交集西部网陕西头条记者从陕西疾控获悉经评估7月17日27日内曾有过湖南省张家界凤凰古城贵州铜仁机场等地旅居史人员属于高风险人群现就该部分人员管控要求通告如下1请立即向当地社区村或疾控机构报告配合落实相应疫情防控措施2为了您和家人的健康安全以及疫情防控大局请广大群众关注官方权威发布不信谣不传谣不造谣积极配合开展疫情防控工作'
	one_hot,label = predict_single(test_text)
	print("测试样本预测标签：",label)

	# evaluate()