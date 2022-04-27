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

# 中青文章库mysql的配置（仅内网地址，服务器上可访问） wx_article_detail
zqkd_article_content={
    "host":"sjxl4n6ub99jbdn4aq0b-rw4rm.rwlb.rds.aliyuncs.com",
    "user":"kd_article",
    "password":"KD!$article1223",
    "database":"zqkd-article"
}

zqkd_wx_feed = {   # 内网
    'host': 'sjxl4n6ub99jbdn4aq0b-rw4rm.rwlb.rds.aliyuncs.com',
    'user': 'big_data',
    'password': 'FDSGSD32DFG!21dDSF',
    'db': 'zqkd-article',
    'charset': 'utf8mb4'
}

zq_wx_feed = {   # 外网
    'host': 'rm-2zef5203kj7og7pujzo.mysql.rds.aliyuncs.com',
    'user': 'kd_article',
    'password': 'KD!$article1223',
    'db': 'zqkd-article',
    'charset': 'utf8mb4'
}

zqkd_article_db = pymysql.connect(**zq_wx_feed)
# zqkd_content_db = pymysql.connect(**zqkd_article_content)



def get_predictions(itemId,sentences):
    # url = 'http://47.94.110.131:9012/polls/category'  # django api路径
    # url = 'http://172.17.2.223:9012/polls/category'  # django 集群内网路径
    # url = 'http://127.0.0.1:9012/polls/category'  # django 集群内网路径
    url = 'http://47.94.110.131:9012/polls/category'  # django 集群外网路径

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
    """去掉html符号"""
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

def main():
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
    id_title_df = pd.read_csv("../data/news/multi_cls/news_data.csv", sep="\t", skiprows=1, header="infer"
                              , names=["id","title"])
    print(id_title_df.head())

    # sql_content = "select id,content from wx_article_detail where id in ({})".format(",".join([str(i['id']) for i in results_title]))
    # results_content = query_mysql(zqkd_content_db,sql_content)
    #
    # id_content_df = pd.DataFrame(results_content)
    # id_content_df['content'] = id_content_df['content'].apply(lambda x:remove_punctuation(contentParser(x)))
    # ttile_content_df = pd.merge(id_title_df,id_content_df)

    # ttile_content_df = title_content_df[["id","title","content","tag_id"]]
    title_content_df = id_title_df[["id","title"]]
    firstCate_list = []
    fp = open("./news_data_cat.txt","a")
    count = 0
    for row in title_content_df.itertuples():
        # print(row)  # row[0] 是index
        itemId = row[1]
        sentence = row[2] # +"__"+ row[3]
        # print("sentence: ",sentence)
        # print(itemId,row[2],sentence[:10])
        cate = get_predictions(itemId, sentence)
        row = str(itemId) + "\t" + cate + "\t" + sentence + "\n"
        fp.write(row)
        count += 1
        print("{}-----".format(count))

    df = pd.read_table("./news_data_cat.txt",sep="\t",header=None,names=["id","cate","title"])
    # print(df)
    df.to_csv("./news_data_cat.csv",sep="\t",index=None,header=["id","cate","title"])

    # print("firstCate_list: ",firstCate_list)


if __name__ == '__main__':
    st1 = time.time()
    main()
    print("耗时： {} ms".format((time.time() - st1)*1000))