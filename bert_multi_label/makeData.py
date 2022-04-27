#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : makeData.py
#@ide    : PyCharm
#@time   : 2022-04-26 17:54:17
'''
from elasticsearch import Elasticsearch, RequestsHttpConnection

class EsUtils():
    def __init__(self):
        es = Elasticsearch(
            ['es-cn-zvp2gm3cb001wzu07.elasticsearch.aliyuncs.com'],
            http_auth=('elastic', 'i@7JFnvz*^uXB0hY'),
            port=9200,
            use_ssl=False
        )
        self.es = es
        self.index = "rec_article_keywords"

    def put_to_es(self,put_json):
        """ res = es.index(index="<YourEsIndex>", doc_type="_doc", id= < YourEsId >, body = {
             "<YourEsField1>": "<YourEsFieldValue1>", "<YourEsField2>": "<YourEsFieldValue2>"})"""
        put_num = 0
        # print("put_json: ",put_json)
        res = self.es.index(index=self.index, doc_type="_doc", id=put_json['item_id'], body=put_json)
        # print("res: ",res)
        put_num = 1
        return put_num

    def query_es(self,query_by):
        query_all = {"match_all": {}}
        query_range = {
            "query": {
                "range": {
                    "age": {
                        "gte": 18,  # >=18
                        "lte": "30"  # <=30
                    }
                }
            }
        }

        resp = self.es.search(index=self.index, query=query_all)
        print("Got %d Hits:" % resp['hits']['total']['value'])
        for hit in resp['hits']['hits']:
            print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])
        pass

def consume_kafka():
    pass

