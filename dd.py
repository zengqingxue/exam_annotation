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



def doIt(num):
	print("Process num is : %s" % num)
	time.sleep(1)
	print('process  %s end' % num)
if __name__ == '__main__':
	# print('mainProcess start')
	# #记录一下开始执行的时间
	# start_time = time.time()
	# #创建三个子进程
	# pool = multiprocessing.Pool(3)
	# print('Child start')
	# for i in range(3):
	# 	pool.apply(doIt,[i])
	# print('mainProcess done time:%s s' % (time.time() - start_time))
	a = ["123",'233','344','edede ']
	b = [str(i) for i in a if str(i).isdigit()]
	print(b)
	# print("1223333".isdigit())
	# print("qqqqqq".isdigit())


