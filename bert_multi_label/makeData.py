#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : makeData.py
#@ide    : PyCharm
#@time   : 2022-04-26 17:54:17
# '''
# from elasticsearch import Elasticsearch, RequestsHttpConnection
#
# class EsUtils():
#     def __init__(self):
#         es = Elasticsearch(
#             ['es-cn-zvp2gm3cb001wzu07.elasticsearch.aliyuncs.com'],
#             http_auth=('elastic', 'i@7JFnvz*^uXB0hY'),
#             port=9200,
#             use_ssl=False
#         )
#         self.es = es
#         self.index = "rec_article_keywords"
#
#     def put_to_es(self,put_json):
#         """ res = es.index(index="<YourEsIndex>", doc_type="_doc", id= < YourEsId >, body = {
#              "<YourEsField1>": "<YourEsFieldValue1>", "<YourEsField2>": "<YourEsFieldValue2>"})"""
#         put_num = 0
#         # print("put_json: ",put_json)
#         res = self.es.index(index=self.index, doc_type="_doc", id=put_json['item_id'], body=put_json)
#         # print("res: ",res)
#         put_num = 1
#         return put_num
#
#     def query_es(self,query_by):
#         query_all = {"match_all": {}}
#         query_range = {
#             "query": {
#                 "range": {
#                     "age": {
#                         "gte": 18,  # >=18
#                         "lte": "30"  # <=30
#                     }
#                 }
#             }
#         }
#
#         resp = self.es.search(index=self.index, query=query_all)
#         print("Got %d Hits:" % resp['hits']['total']['value'])
#         for hit in resp['hits']['hits']:
#             print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])
#         pass
#
# def consume_kafka():
#     pass
mapping_dict = {}
aa = ["美文","健康","生活","美食","三农","搞笑","育儿","教育","娱乐","科技","时尚","文化","财经","宠物","旅游","汽车","军事","科学","职场","体育","动漫","历史","国际","房产","时政","游戏","社会","科技","财经","运势"]
cate_26 = ["职场","运势","育儿","娱乐","游戏","文化","体育","时政","时尚","社会","情感","汽车","美食","旅游","历史","科学","科技","军事","教育","健康","家居","国际","搞笑","房产","动漫","财经"]
# mapping_dict['财经'] = '三农'
# mapping_dict['职场'] = '美文'
# mapping_dict['情感'] = '美文'
# mapping_dict['文化'] = '美文'
# mapping_dict['育儿'] = '生活'
# mapping_dict['娱乐'] = '生活'
# mapping_dict['时尚'] = '生活'
# mapping_dict['美食'] = '生活'
# mapping_dict['旅游'] = '生活'
# mapping_dict['家居'] = '生活'
# mapping_dict['宠物'] = '社会'
# mapping_dict['宠物'] = '家居'
mapping_dict['三农'] = ['财经','时政']
mapping_dict['美文'] = ['职场','情感','文化']
mapping_dict['生活'] = ['时尚','美食','旅游','家居']
mapping_dict['宠物'] = ['社会','家居']

print(set(aa).difference(set(cate_26)))
print(mapping_dict)
