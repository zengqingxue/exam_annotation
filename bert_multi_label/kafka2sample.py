#!/user/bin/env python
# -*- coding: utf-8 -*-
'''
@project : exam_annotation
@author  : zengqingxue
#@file   : kafka2sample.py
#@ide    : PyCharm
#@time   : 2022-04-29 16:49:19
'''
# from kafka import KafkaConsumer
import time
topic_name = 'zqkdnews_update'
bootstrap_servers = ["172.17.24.5:9092","172.17.24.3:9092","172.17.24.2:9092","172.17.24.4:9092"]
auto_offset = "earliest"
def consume_demo():
    consumer = KafkaConsumer(topic_name,
                             bootstrap_servers=bootstrap_servers)

    for message in consumer:
        print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                              message.offset, message.key,
                                              message.value))
def consume_offset():
    from kafka import KafkaConsumer
    from kafka.structs import TopicPartition

    consumer = KafkaConsumer(topic_name,
                             auto_offset_reset=auto_offset,
                             bootstrap_servers=bootstrap_servers)

    print(" 获取test主题的分区信息: \n",consumer.partitions_for_topic(topic_name)) # 获取test主题的分区信息
    print("获取主题列表: \n",consumer.topics())  # 获取主题列表
    print("获取当前消费者订阅的主题: \n",consumer.subscription())  # 获取当前消费者订阅的主题
    print("获取当前消费者topic、分区信息：\n",consumer.assignment())  # 获取当前消费者topic、分区信息
    print("获取当前消费者可消费的偏移量: \n",consumer.beginning_offsets(consumer.assignment()))  # 获取当前消费者可消费的偏移量
    # consumer.seek(TopicPartition(topic=topic, partition=0), 0)  # 重置偏移量，从第5个偏移量消费
    # consumer.seek(TopicPartition(topic=topic, partition=1), 0)  # 重置偏移量，从第5个偏移量消费
    # consumer.seek(TopicPartition(topic=topic, partition=2), 0)  # 重置偏移量，从第5个偏移量消费
    # consumer.seek(TopicPartition(topic=topic, partition=3), 0)  # 重置偏移量，从第5个偏移量消费
    # consumer.seek(TopicPartition(topic=topic, partition=4), 0)  # 重置偏移量，从第5个偏移量消费
    # consumer.seek(TopicPartition(topic=topic, partition=5), 0)  # 重置偏移量，从第5个偏移量消费

    fp = open("./cate_checked.txt")
    for message in consumer:
        topic = message.topic.decode("utf-8")
        partition =message.partition.decode("utf-8")
        offset = message.offset.decode("utf-8")
        key = message.key.decode("utf-8")
        value =  message.value.decode("utf-8")
        if key.find("category") != -1:
            if key is not None and key.find("category") != -1:
                print("topic,partition,offset=,key,value: {} {} {} {} {}".format(topic, partition,offset, key,value))
                fp.write(key + "\t" + value + "\n")

if __name__ == '__main__':
    consume_offset()
