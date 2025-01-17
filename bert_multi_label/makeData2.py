#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : makeData2.py
#@ide    : PyCharm
#@time   : 2022-04-26 19:57:46
'''
"""给预测服务发post请求，获取预测结果"""
import json

import pandas as pd
import requests
import pymysql
import time
import re
import os
import time
import random
import sys
from multiprocessing import Pool
from multiprocessing import process
import pymysql.cursors
from dbutils.pooled_db import PooledDB
import argparse

# 中青文章库mysql的配置（仅内网地址，服务器上可访问） wx_article_detail
zqkd_article_content = {
    "host": "sjxl4n6ub99jbdn4aq0b-rw4rm.rwlb.rds.aliyuncs.com",
    "user": "kd_article",
    "password": "KD!$article1223",
    "database": "zqkd-article",
    "charset":'utf8mb4'
}

zqkd_wx_feed = {  # 内网
    'host': 'sjxl4n6ub99jbdn4aq0b-rw4rm.rwlb.rds.aliyuncs.com',
    'user': 'big_data',
    'password': 'FDSGSD32DFG!21dDSF',
    'db': 'zqkd-article',
    'charset': 'utf8mb4'
}

zq_wx_feed = {  # 外网
    'host': 'rm-2zef5203kj7og7pujzo.mysql.rds.aliyuncs.com',
    'user': 'kd_article',
    'password': 'KD!$article1223',
    'db': 'zqkd-article',
    'charset': 'utf8mb4'
}


# zqkd_article_db = pymysql.connect(**zq_wx_feed)

# pool = PooledDB(pymysql, 12, **sql_db["pro_env"]["recommend_db"], setsession=['SET AUTOCOMMIT = 1'])
# pool = PooledDB(pymysql, 12, **zq_wx_feed, setsession=['SET AUTOCOMMIT = 1'])
# recommend_db = pool.connection()

def get_predictions(itemId, sentences):
    # url = 'http://47.94.110.131:9012/polls/category'  # django api路径
    # url = 'http://172.17.2.223:9012/polls/category'  # django 集群内网路径
    # url = 'http://127.0.0.1:9012/polls/category'  # django 集群内网路径
    # url = 'http://47.94.110.131:9012/polls/category'  # django 集群外网路径
    url = 'http://172.17.2.202:9012/polls/category'  # django 集群内网网路径

    parms = {}
    parms["itemId"] = str(itemId)
    parms["sentences"] = sentences

    headers = {  # 请求头 是浏览器正常的就行 就这里弄了一天 --！
        'User-agent': 'none/ofyourbusiness',
        'Spam': 'Eggs'
    }
    # print(parms)
    resp = requests.post(url, data=parms, headers=headers)  # 发送请求
    text = " ".join([cat for cat in json.loads(resp.text)])
    # print("itemId,text,sentence[:30]: ",itemId,text,str(sentences)[:30])
    return text

    # print(json.loads(text))


def read_csv_postcate(inputfile,outputfile):
    df = pd.read_csv(inputfile,sep="\t",header=None,index_col=None,names=["id","label","title_content"])
    count = 0
    fp = open(outputfile,"a")
    for row in df.itertuples():
        itemId = row[1]
        sentence = row[3]
        cate = get_predictions(itemId, sentence)
        # row = str(itemId) + "\t" + cate + " " + row[2] + "\t" + sentence + "\n"
        row = str(itemId) + "\t" + cate.split(" ")[0] + "\t" + sentence + "\n"
        fp.write(row)
        count += 1
        print("{} -----".format(count))
    fp.close()

def contentParser(content):
    """去掉html符号 包括短路奥德换行符\n"""
    line = ""
    try:
        rehtml = re.compile(r'<[^>]+>', re.S)
        line = rehtml.sub('', content.replace("\n", ""))
    except:
        print("line: ", content)
    return line

def remove_punctuation(line):
    """去掉空格标点符号"""
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line

def remove_html_punc(content):
    return remove_punctuation(contentParser(content))

def query_mysql(db, sql):
    rows = []
    with db.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
        try:
            num = cursor.execute(sql)
            if num > 0:
                rows = cursor.fetchall()
                # print("本次查询到数量为： ", num)
            else:
                # print("本批次查询mysql返回全为空。。。。。。")
                pass
        except Exception as e:
            print("sql: {}   执行出问题了。。。。。".format(sql))
            print(e)
        finally:
            pass
    return rows


import math

def query_title_content_tagname(recommend_db,tagname,zqkd_content_db,label):
    with recommend_db.cursor(pymysql.cursors.DictCursor) as cursor:
        query_sql = """SELECT id,title FROM wx_feed where tagname in (%s)  and `type`=1"""%",".join(["'%s'"] * len(tagname.split(",")))%(tuple(tagname.split(",")))
        # query_sql = """SELECT id,title FROM wx_feed where tag_id in (%s)  and `type`=1"""%(tagname)
        print("query_sql: ",query_sql)
        row_num = cursor.execute(query_sql)
        if row_num:
            id_title = cursor.fetchall()
            # id_title_dict_list.extend(id_title)
            id_title_df = pd.DataFrame(id_title)
            id_title_df['title'] = id_title_df['title'].apply(lambda x: remove_html_punc(x))

            # 查询content
            id_list = id_title_df['id'].values.tolist()
            print("id_list: ", len(id_list))
            print("id_list[:10]: ", id_list[:10])
            sql_content = "select id,content from wx_article_detail where id in ({})".format(
                ",".join([str(i) for i in id_list]))

            results_content = query_mysql(zqkd_content_db, sql_content)
            print("results_content: ",results_content[:2])
            id_content_df = pd.DataFrame(results_content)
            print("id_content_df: ",id_content_df)
            id_content_df['content'] = id_content_df['content'].apply(lambda x: remove_punctuation(contentParser(x)))
            title_content_df = pd.merge(id_title_df, id_content_df)
            title_content_df['title_content'] = title_content_df['title'].str.cat(title_content_df["content"],sep="__")
            title_content_df['label'] = label
            title_content_df =  title_content_df[['id','label','title_content']]
            title_content_df.to_csv("../data/news/multi_cls/{}.csv".format(label), sep="\t", header=None, index=None)

def query_batch(recommend_db):
    """分页查询msyqL"""
    # db = pymysql.connect(**zq_wx_feed)
    with recommend_db.cursor(pymysql.cursors.DictCursor) as cursor:
        # count_sql = "SELECT count(1) as count FROM wx_feed"
        # query_num = cursor.execute(count_sql)
        # num = 0
        # if query_num :
        #     num = cursor.fetchall()[0]['count']
        # print("总共识别条数为： ", num)
        num = 3000000
        batch_size = 30000  # 分钟集增量更新用
        # id_title_dict_list = []
        for i in range(0, math.ceil(num / batch_size)):
            # for i in range(0,2): # 测试时用
            step = i * batch_size
            query_sql = "SELECT id,title FROM wx_feed order by crawl_time desc limit %s,%s" % (step, batch_size)
            row_num = cursor.execute(query_sql)
            if row_num:
                id_title = cursor.fetchall()
                # id_title_dict_list.extend(id_title)
                id_title_df = pd.DataFrame(id_title)
                id_title_df['title'] = id_title_df['title'].apply(lambda x:remove_html_punc(x))
                id_title_df.to_csv("./id_title_{}.csv".format(i), sep="\t", header=None, index=None)
                print(i)

def query_content():
    id_title_df = pd.read_csv("../data/news/traindata/id_title_2.csv", sep="\t", skiprows=0, header="infer"
                              , names=["id", "title"])
    id_list = id_title_df['id'].values.tolist()

def main1():
    # sql_title = "SELECT id,tag_id,title,abstract from wx_feed where id in (42870841)"
    # sql_title = "SELECT id,tag_id,title,abstract from wx_feed where crawl_time>UNIX_TIMESTAMP(DATE_SUB(now(),INTERVAL 1 hour)) limit 100"
    #    sql_title = """
    #    (SELECT id,title from wx_feed  where crawl_time>UNIX_TIMESTAMP(DATE_SUB(now(),INTERVAL 30 day)))
    # UNION all (SELECT id,title from wx_feed_cold where crawl_time>UNIX_TIMESTAMP(DATE_SUB(now(),INTERVAL 30 day)))
    #    """
    #    results_title = query_mysql(zqkd_article_db,sql_title)
    #    id_title_df = pd.DataFrame(results_title)
    #    id_title_df.to_csv("./news_data.csv",sep="\t",index=None)
    #
    id_title_df = pd.read_csv("../data/news/traindata/id_title_2.csv", sep="\t", skiprows=0, header="infer"
                              , names=["id", "title"])
    print(id_title_df.head())

    # title_content_df = title_content_df[["id","title","content","tag_id"]]
    title_content_df = id_title_df[["id", "title"]]
    firstCate_list = []
    fp = open("./news_data_cat.txt", "a")
    count = 0
    for row in title_content_df.itertuples():
        # print(row)  # row[0] 是index
        itemId = row[1]
        sentence = row[2]  # +"__"+ row[3]
        # print("sentence: ",sentence)
        # print(itemId,row[2],sentence[:10])
        cate = get_predictions(itemId, sentence)
        row = str(itemId) + "\t" + cate + "\t" + sentence + "\n"
        fp.write(row)
        count += 1
        print("{}-----".format(count))

    df = pd.read_table("./news_data_cat.txt", sep="\t", header=None, names=["id", "cate", "title"])
    df.to_csv("./news_data_cat.csv", sep="\t", index=None, header=["id", "cate", "title"])

def worker(num, file, zqkd_content_db):
    print("开始file: ", file)
    id_title_df = pd.read_csv(file, sep="\t", header=None, names=["id", "title"])
    id_list = id_title_df['id'].values.tolist()
    print("id_list: ", len(id_list))
    print("id_list[:10]: ", id_list[:10])
    sql_content = "select id,content from wx_article_detail where id in ({})".format(
        ",".join([str(i) for i in id_list]))
    results_content = query_mysql(zqkd_content_db, sql_content)
    id_content_df = pd.DataFrame(results_content)
    id_content_df['content'] = id_content_df['content'].apply(lambda x: remove_punctuation(contentParser(x)))
    title_content_df = pd.merge(id_title_df, id_content_df)
    title_content_df['label']
    title_content_df = title_content_df[["id", "title", "content"]]
    # fp = open("./news_data_cat_{}.txt".format(num), "a")
    # count = 0
    # for row in title_content_df.itertuples():
    #     itemId = row[1]
    #     sentence = row[2] + "__" + row[3]
    #     cate = get_predictions(itemId, sentence)
    #     row = str(itemId) + "\t" + cate + "\t" + sentence + "\n"
    #     fp.write(row)
    #     count += 1
    #     print("{} {} -----".format(os.getpid(), count))


def parse_arg():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mc1', dest='mc1',
    #                     type=str,
    #                     help='please input model category \n e.g. : python3 Vearch.py --mc2 0')
    # parser.add_argument('--mc2', dest='mc2',
    #                     type=str,
    #                     help='please input model category \n e.g. : python3 Vearch.py --mc2 10')
    # parser.add_argument('--output_dir', dest='output_dir',
    #                     type=str,
    #                     help='please input model category \n e.g. : python3 Vearch.py --mc2 10')
    # parser.add_argument('--file_num', dest='file_num',
    #                     type=str,
    #                     help='please input model category \n e.g. : python3 Vearch.py --file_num 10')
    parser.add_argument('--tagname', dest='tagname',
                        type=str,
                        help='please input model category \n e.g. : python3 Vearch.py --tagname 10')
    parser.add_argument('--label', dest='label',
                        type=str,
                        help='please input model category \n e.g. : python3 Vearch.py --label 10')
    args = parser.parse_args()
    return vars(args)


def main():
    file_list = os.listdir("./")
    file_list = [file_name for file_name in file_list if file_name.find("id_title") != -1]
    argparams = parse_arg()
    mc1 = int(argparams["mc1"])
    mc2 = int(argparams["mc2"])
    file_list = sorted(file_list)[mc1:mc2]
    print("file_list的长度 ", file_list)
    pool = PooledDB(pymysql, 12, **zqkd_article_content, setsession=['SET AUTOCOMMIT = 1'])
    # zqkd_content_db = pymysql.connect(**zqkd_article_content)
    zqkd_content_db = pool.connection()
    for num, file in enumerate(file_list):
        worker(num, file, zqkd_content_db)


def query_by_id(zqkd_content_db):
    id_list = [36080068]
    sql_content = "select id,content from wx_article_detail where id in ({})".format(
        ",".join([str(i) for i in id_list]))
    results_content = query_mysql(zqkd_content_db, sql_content)
    content = remove_html_punc(results_content[0]['content'])
    print("results_content: ",results_content)
    print("results_content: ", content)
    return content

def read_csv_drop_duplicates(inputfile,outputfile):
    df = pd.read_csv(inputfile,sep="\t",header=None,index_col=None,names=["id","label","title_content"])
    # df = df.drop_duplicates(subset=['id', 'label'])
    df1 = df.groupby(['id','title_content'])['label'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
    print(df1)
    df1['label2'] = df1.apply(lambda x: " ".join(set(x.label.split(" "))),axis=1)
    print(df1)
    df = df1[['id','label2','title_content']]
    # df = df.drop_duplicates(subset=['id', 'label']).reset_index(drop=True)
    df.to_csv(outputfile,header=None,index=None,sep="\t")

def merge_duplic_sample_label(df):
    # df =  pd.DataFrame({
    #      "id":[1,2,3,1,2],
    #      "title": ["1", "2", "3", "1", "2"],
    #      'label': ['A',"B","C","C","D"]
    #  })
    # print(df)
    df_label_merge = df.groupby('id')['label'].apply(lambda x: x.str.cat(sep=' ')).reset_index()
    print("df_label_merge: ",df_label_merge)
    df_label_merge.to_csv("./finally_samples.csv", sep="\t", header=None, index=None)


if __name__ == '__main__':
    # inputfile, outputfile = './news_33.csv',"./news_33.csv.0"
    # read_csv_drop_duplicates(inputfile, outputfile)
    itemId = "11111"
    # sentences = "同行密接人员手机自查最快只要3秒钟打开方式戳这里__近日全国多地疫情出现反复武汉今日新增确诊病例7例湖北疫情防控措施全面升级目前国家政务服务平台已在支付宝上线了同行密接人员自查服务武汉用户可上支付宝搜同行自查免费查询或者在支付宝小程序国家政务服务平台里点击同行密接人员自查最快只需3秒钟查一查更安心同行自查服务由国家政务服务平台和卫健委联合提供14天内乘坐过飞机或者火车等公共交通工具出行的朋友可放心查询为了方便您下次更快速找到该服务可以点击右上角将小程序添加到支付宝首页此外如果你想知道自己所在地区或者出行目的地地区的疫情风险等级国务院客户端已在支付宝上线了查询服务您可搜疫情等级查看需要注意的是湖北近日将加大对各公共场所健康码绿码查验工作出入火车站机场医院商场等公共场所或者搭乘公交地铁均需出示健康码为方便快速找到健康码建议武汉市民将健康码添加到支付宝首页可快速打开核验长江日报出品采写记者张珺编辑叶凤校对熊琳琳"
    sentences = "断卡行动交易流水达30亿元涉案26人宁夏一起杀猪盘诈骗案件宣判__历时8个月中宁县公安局成功侦破了一起财产损失最大的杀猪盘诈骗案件打掉洗钱团伙2个抓获犯罪嫌疑人26名扣押作案手机50余部涉案银行卡电话卡400余张冻结涉案资金120余万元为受害群众挽回损失近50余万元26名犯罪嫌疑人全部判处实刑2020年3月26日辖区群众陈某到中宁县公安局报警称2020年3月5日其在某网站上遇到一男子后发展为恋爱关系之后按照对方介绍的方法在一个网站上注册进行投资投资后很长时间无法提现逐渐意识到被骗共累计损失69余万元该案是当时中宁县发生电信网络诈骗案件财产损失最大的一起杀猪盘诈骗案件案发后中宁县公安局党委高度重视立即成立工作专班对该案的资金流进行梳理经过分析研判顺线追踪确定了位于河南云南的两个专门从事洗钱的犯罪团伙并通过进一步侦查逐渐查清了这两个团伙的活动轨迹组织结构和成员身份信息之后经过多次会商研究专案组果断决定收网参战民警克服异地抓捕对地形环境不熟悉犯罪团伙流动性大审讯羁押等一系列困难昼夜奋战辗转河南山东云南广州等地成功打掉了这两个为诈骗犯罪分子洗钱的团伙共抓获犯罪嫌疑人26名现场扣押作案手机50余部涉案银行卡电话卡400余张据了解该团伙成员和境外电信网络诈骗团伙勾结长期大量收购手机卡银行卡和转账U盾等配套工具为境外电信网络诈骗团伙实施违法犯罪提供支付结算工具两个犯罪团伙涉及的银行卡交易流水高达数亿元该案于2021年7月初由中宁县人民法院依法作出判决涉案26人分别被判处八个月至二年四个月不等的有期徒刑是中宁县宣判的首个特大网络杀猪盘类诈骗案件也是目前为止获刑人数最多的电信网络诈骗案件中宁县公安局将继续深入开展断卡行动不断提升对电信网络诈骗犯罪的打击力度全力遏制电信网络诈骗犯罪的高发多发态势为维护社会大局稳定保护人民群众生命财产安全作出更大的贡献声明本文来源平安中宁在此致谢"
    print(get_predictions(itemId, sentences))

    # region read_csv_postcate
    # args = parse_arg()
    # inputfile = args['tagname']
    # outputfile = args['label']
    # prefix = "/root/zengqingxue/exam_annotation/data/news/multi_cls/"
    # inputfile, outputfile = prefix+inputfile,prefix+outputfile
    # read_csv_postcate(inputfile, outputfile)
    # endregion

    # # region 根据tagname抽取样本
    # argparams = parse_arg()
    # pool = PooledDB(pymysql, 12, **zqkd_wx_feed, setsession=['SET AUTOCOMMIT = 1'])
    # recommend_db = pool.connection()
    # pool = PooledDB(pymysql, 12, **zqkd_article_content, setsession=['SET AUTOCOMMIT = 1'])
    # zqkd_content_db = pool.connection()
    # tagname = argparams["tagname"]
    # label = argparams["label"]
    # # prefix = "/root/zengqingxue/exam_annotation/data/news/multi_cls/"
    # # outputfile = prefix + label + ".csv"
    #
    # query_title_content_tagname(recommend_db,tagname,zqkd_content_db,label)
    # # endregion

    # region others
    # df= pd.read_csv("./tmp0/id_title_27.csv",sep="\t",header=None,index_col=None,names=["id,title"])
    # import numpy as np
    # df = pd.DataFrame({"id":[11,22,33,44,555],
    #               "title": ["222", np.nan,3344.44,None,np.NaN]
    #               })
    # df['title'] = df['title'].astype("string")
    # df = df.where(pd.notnull(df),"11")
    # print(df[df['title']=='11'])
    # print(df)
    # df = df.apply(lambda x:x['title']==np.nan,axis=1)
    # print(df.isnull)
    # print(df.dtypes)
    # print(  df.isnull())
    # print(remove_html_punc(".我的世界ee__....."))
    # query_sql = "SELECT id,title FROM wx_feed where id=%s" % (36080068)
    # results_title = query_mysql(recommend_db, query_sql)
    # title = results_title[0]['title'].replace("\n",'')
    # print(title)
    # pool = PooledDB(pymysql, 12, **zqkd_article_content, setsession=['SET AUTOCOMMIT = 1'])
    # # zqkd_content_db = pymysql.connect(**zqkd_article_content)
    # zqkd_content_db = pool.connection()
    # content = query_by_id(zqkd_content_db)
    # print(title +"__"+ content)
    # st1 = time.time()
    # main()
    # # query_batch(recommend_db)
    # print("耗时： {} ms".format((time.time() - st1)*1000))
    # i = 0
    # while True:
    #     print("nohup python makeData.py --mc1={}".format(str(i)),"--mc2={}"
    #           .format(str(i+10)),">> nohup_{}_1.out 2>&1  &".format(str(i+10)))
    #     i += 10
    #     if i >= 100:
    #         break

    #     """
    #     (recommend) [root@iZ2ze260sesa08m8bnslzbZ zengqingxue1]# ps axuw | grep makeData.py
    # root      5294  0.0  0.0 112832  1008 pts/1    S+   20:31   0:00 grep --color=auto makeData.py
    # root     11743 12.9  0.4 1276004 745624 pts/1  Sl   20:16   1:58 python makeData.py --mc1=0 --mc2=10
    # root     44875  7.2  0.2 972820 442260 pts/1   Sl   20:26   0:21 python makeData.py --mc1=10 --mc2=20
    # root     44879  6.7  0.1 724432 194128 pts/1   Sl   20:26   0:19 python makeData.py --mc1=50 --mc2=60
    # root     44880  7.1  0.1 796408 267868 pts/1   Sl   20:26   0:20 python makeData.py --mc1=60 --mc2=70
    # root     44883  7.2  0.2 896128 365512 pts/1   Sl   20:26   0:21 python makeData.py --mc1=70 --mc2=80
    # root     44884  7.1  0.1 830560 300164 pts/1   Sl   20:26   0:20 python makeData.py --mc1=80 --mc2=90
    # root     44888  7.2  0.2 864760 334236 pts/1   Sl   20:26   0:21 python makeData.py --mc1=90 --mc2=100
    #     """
    # endregion