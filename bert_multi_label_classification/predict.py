#! -*- coding: utf-8 -*-
import pickle
from tqdm import tqdm
from sklearn.metrics import hamming_loss, classification_report
from train import *
from loguru import logger
logger.add('./logs/my.log', format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> - {module} - {function} - {level} - line:{line} - {message}", level="INFO",rotation='00:00',retention="3 day")
from config import Config

config = Config()
mlb_path = config.mlb_path
threshold = config.prob_threshold
test_data = config.test_data
mlb = pickle.load(open(mlb_path,'rb'))
logger.info(mlb.classes_.tolist())


def predict_single(test_text):
	token_ids, segment_ids = tokenizer.encode(test_text, maxlen=maxlen)
	pred = model.predict([[token_ids], [segment_ids]])
	logger.info("pred[0]为: {}",pred[0])

	label_index = np.where(pred[0]>threshold)[0] # 取概率值大于阈值的 onehot 向量索引, [12,34]
	# 根据prob 排序 label_index
	label_prob = {pred[0][i]:i for i in label_index}
	rerank = sorted(label_prob.items(), key=lambda x: x[0], reverse=True)
	label_index = [i[1] for i in rerank]

	logger.info("label_index为: {}", label_index)
	labels = [mlb.classes_.tolist()[i] for i in label_index]
	one_hot_label = np.where(pred[0]>threshold,1,0) # [[0,0,1,0,0,..],[0,0,1,0,1,..]]
	return one_hot_label, ' '.join(labels)


def evaluate():
    test_x,test_y = load_data(test_data)
    true_y_list = mlb.transform(test_y)

    pred_y_list = []
    for text in tqdm(test_x):
        pred_y, pred_label = predict_single(text)
        pred_y_list.append(pred_y)
    # F1值
    logger.info("F1值为：\n {} ",classification_report(true_y_list, pred_y_list, digits=4,target_names=mlb.classes_.tolist()))#


if __name__ == '__main__':
	# test_text = "这几款早餐做法简单营养美味学会了家人的早餐不会愁__经典家常鸡蛋香菇油菜大蒸包宝宝辅食食谱牛油果香蕉卷鸡蛋饼竟然可以这样做给宝宝的早餐也可以如此创意洋葱圈蔬菜蛋饼原味戚风蛋糕戚风蛋糕黄金玉米烙山药红豆糕肉酱意大利面铜锣烧金龙鱼河套平原雪花粉面粉用白糖香满园高活性干酵母面粉用温水鸡蛋鲜香菇油菜葱花油盐芝麻香油牛油果香蕉吐司鸡蛋鸡蛋配方奶低筋面粉JUSTEgg皆食得植物蛋玉米淀粉胡萝卜玉米芦笋洋葱白胡椒粉盐蛋黄细砂糖牛奶玉米油盐低筋面粉蛋白细砂糖低筋面粉牛奶调和油鸡蛋细砂糖蛋清细砂糖蛋黄柠檬玉米粒干淀粉白砂糖山药红豆白糖意大利面肉馅洋葱胡萝卜西芹西红柿番茄酱意大利综合香料蒜黑胡椒碎糖芝士粉橄榄油盐鸡蛋面粉牛奶糖泡打粉红豆沙"
	test_text = '平度潜力盘流出龙腾华城三居室均价仅07万首付仅29万起__房源详情城市青岛小区龙腾华城面积建面116平户型3室2厅1厨1卫价格85万楼层中楼层朝向南北装修毛坯推荐理由每平米较小区均价低1479元宽敞明亮大三居孩子老人一同居住阖家居住老人孩子搭乘电梯不拥挤核心卖点本房源为宽敞明亮大三居适合孩子老人一同居住全明格局采光通风俱佳开发商原始毛坯房适合对房屋装修有要求的进行重新设计装修有电梯适合老人和孩子生活方便快捷'
	# test_text = '置业城阳必看价格喜人每平16万青特小镇三居室首付59万__房源详情城市青岛小区青特小镇面积建面104平户型3室2厅1厨1卫价格170万楼层中楼层朝向南北装修毛坯推荐理由房龄短舒适三居室容纳全家人的幸福电梯配套极大提高生活品质核心卖点本房源为舒适三居室容纳全家人的幸福南北通透空气对流快采光效果佳冬暖夏凉毛坯房可随心装修改造空间大电梯房大大提高生活品质建于2017房龄很新'
	# test_text = '南京疫情外溢6省11市感染至少170人病例情况一图读懂__自7月20日南京禄口机场检出9例阳性病例以来截至7月27日24时南京本轮疫情累计报告155例新冠感染者根据已公布的病例详情有90余人都是该机场的工作人员66人系从事保洁工作另外还涉及到机场的司机地勤辅警水电工等加上广东中山辽宁沈阳安徽和县江苏宿迁四川绵阳安徽芜湖广东珠海四川泸州四川成都辽宁大连湖南常德6省11市报告的相关联感染者南京此轮疫情感染者总计至少达170人'
	# test_text = '陕西疾控这些人员属于高风险人群请立即报告__西部网讯记者马晴茹因大连成都常德等地连日来报告的多例确诊病例和无症状感染者均在湖南省张家界等地有时间和空间上的轨迹交集西部网陕西头条记者从陕西疾控获悉经评估7月17日27日内曾有过湖南省张家界凤凰古城贵州铜仁机场等地旅居史人员属于高风险人群现就该部分人员管控要求通告如下1请立即向当地社区村或疾控机构报告配合落实相应疫情防控措施2为了您和家人的健康安全以及疫情防控大局请广大群众关注官方权威发布不信谣不传谣不造谣积极配合开展疫情防控工作'
	one_hot,label = predict_single(test_text)
	logger.info("测试样本预测标签： {} \n one_hot为: {}",label,one_hot)

	# evaluate()
