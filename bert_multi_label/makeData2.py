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
    "database": "zqkd-article"
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

            # print("sql_content: ",sql_content)
            results_content = query_mysql(zqkd_content_db, sql_content)
            print("results_content: ",results_content[:2])
            id_content_df = pd.DataFrame(results_content)
            print("id_content_df: ",id_content_df)
            id_content_df['content'] = id_content_df['content'].apply(lambda x: remove_punctuation(contentParser(x)))
            title_content_df = pd.merge(id_title_df, id_content_df)
            title_content_df['title_content'] = title_content_df['title'].str.cat(title_content_df["content"],sep="__")
            title_content_df['label'] = label
            title_content_df =  title_content_df[['id','label','title_content']]
            title_content_df.to_csv("./{}.csv".format(label), sep="\t", header=None, index=None)


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
    argparams = parse_arg()
    pool = PooledDB(pymysql, 12, **zqkd_wx_feed, setsession=['SET AUTOCOMMIT = 1'])
    recommend_db = pool.connection()
    pool = PooledDB(pymysql, 12, **zqkd_article_content, setsession=['SET AUTOCOMMIT = 1'])
    zqkd_content_db = pool.connection()
    tagname = argparams["tagname"]
    label = argparams["label"]
    query_title_content_tagname(recommend_db,tagname,zqkd_content_db,label)

    # s ="""
    # <article><p>种鸽近亲配对的方式下，很多鸽友都说后代的鸽子有的活力比较差。平时看起来很安静，虽然说算不上是死气沉沉，但是确实也是比较没有活力。这个也算是近亲回血的时候，后代鸽子的一个比较常见的现象。那么我们鸽友平时又难免需要让鸽子近亲回血配对，可以怎么来提升信鸽的回血后代的活力呢？</p>
    # <div class="pgc-img">
    # <img width="640" height="480" data-width="640" data-height="480" data-tt="http://res.youth.cn/img-detail/8ab264244405f2c60be13bce4045afd1:640:480.jpg"></div>
    # <p>第一，选择活力比较好的，健康的，没有任何退化的鸽子来近亲配对。<span class="entity-word" data-gid="970214">回血鸽</span>虽然可能出现后代活力较差的现象，但是这个现象也不是一定会出现。如果回血的鸽子各方面条件都比较好的话，后代鸽子也很少会出现活力下降或者是存在什么退化问题的现象。</p>
    # <div class="pgc-img">
    # <img width="640" height="676" data-width="640" data-height="676" data-tt="http://res.youth.cn/img-detail/0ae30d053c54ba7c686e44878e75355b:640:676.jpg"></div>
    # <p>第二，有的时候会让鸽子进行多重近亲配对，这个时候我们鸽友还可以注意一下，如果说鸽子确实是出的后代的活力下降比较多，那么这个时候就不要近亲配对了，可以让鸽子杂交配对之后再来近亲配对，或者是远亲配对之后再来近亲配对，这样一来问题就少一点。</p>
    # <div class="pgc-img">
    # <img width="640" height="480" data-width="640" data-height="480" data-tt="http://res.youth.cn/img-detail/737babc2de5411e994b17c145bac5f9e:640:480.jpg"></div>
    # <p>第三，对于偶尔出现的活力下降的情况，可以针对性进行淘汰。也就是说，如果鸽子有的出现活力下降的情况，有的完全正常，那么不正常的鸽子可以进行淘汰。这个问题在近亲配对的时候完全可以根据情况来选择一些鸽子进行淘汰，毕竟有的时候退化等问题并不是说所有后代都出现，有时候概率还是比较低的一个情况。</p></article>
    # """
    # print(remove_html_punc(s))

    # print(",".join(['%s']*3))
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
