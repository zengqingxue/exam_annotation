#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : dd.py
#@ide    : PyCharm
#@time   : 2022-04-27 18:35:31
'''
import time
import multiprocessing

#
#
# def doIt(num):
# 	print("Process num is : %s" % num)
# 	time.sleep(1)
# 	print('process  %s end' % num)
# if __name__ == '__main__':
# 	# print('mainProcess start')
# 	# #记录一下开始执行的时间
# 	# start_time = time.time()
# 	# #创建三个子进程
# 	# pool = multiprocessing.Pool(3)
# 	# print('Child start')
# 	# for i in range(3):
# 	# 	pool.apply(doIt,[i])
# 	# print('mainProcess done time:%s s' % (time.time() - start_time))
# 	a = ["123",'233','344','edede ']
# 	b = [str(i) for i in a if str(i).isdigit()]
# 	print(b)
# 	# print("1223333".isdigit())
# 	# print("qqqqqq".isdigit())

import pandas as pd

def get_only():
    df = pd.read_csv("./news_26.csv",sep="\t",header=None,index_col=None,names=['id','label','content'])
    print(df)
    df = df.sample(frac=1)
    df.dropna(inplace=True)
    print(df)
    df['label_new']  = df.apply(lambda x:x.label.split(" ")[0],axis=1)
    df = df[['id','label_new','content']]
    df.to_csv("./news_26_onlyOne.csv",sep="\t",header=None,index=None)
    label_new = df['label_new'].tolist()
    print("label_new的长度为:",len(label_new))
    label_new_set = set(label_new)
    print("label_new_set: ",label_new_set)

def split_26cate():
    df = pd.read_csv("./news_26_onlyOne.csv",header=None,index_col=None,names=['id','label','content'])
    cate_26 = ["职场", "运势", "育儿", "娱乐", "游戏", "文化", "体育", "时政", "时尚", "社会", "情感", "汽车", "美食", "旅游", "历史", "科学", "科技",
               "军事", "教育", "健康", "家居", "国际", "搞笑", "房产", "动漫", "财经"]
    for i in cate_26:
        df_1 = df[df['label']==i]
        df_1.to_csv("./ft_cate/{}.csv".format(i), header=None, index=None, sep="\t")

import pandas as pd
def get_cha():
    df1 = pd.read_csv("./美文.csv", sep="\t", header=None, index_col=None, names=["id", "label", "title_content"])
    df2 = pd.read_csv("./non_meiwen_all.csv", sep="\t", header=None, index_col=None, names=["id", "label", "title_content"])
    set_diff_df = pd.concat([df1, df2, df2]).drop_duplicates(subset=['id'],keep=False)
    set_diff_df.to_csv("./美文.csv.new",header=None,index=None,sep="\t")
    print("df1 df2 set_diff_df 的形状分别为： {} {} {}".format(df1.shape,df2.shape,set_diff_df.shape))
    print(set_diff_df)

if __name__ == '__main__':
    get_cha()
    # get_cha()
    p = 220000
    j = 1
    for i in range(1,12):
        print("sed -n '{},{}p' news_26.csv > news_26.csv.{}".format(j,p,i))
        j += 220000
        p += 220000




