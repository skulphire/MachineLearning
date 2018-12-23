import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import datetime
import sys

nclasses = 2 # 0=neg, 4= pos? # 1 or 0 now

batch_size = 128
total_batches = 'n'

x = tf.placeholder('float')
y = tf.placeholder('float')

# distribution
# 1 worker, 2 Ps
cluster = tf.train.ClusterSpec({'worker':['10.10.1.142:2221'],'ps':['10.10.1.140:2222','10.10.1.141:2223']})
workerStr = '/job:worker/task:'
jobType = sys.argv[1]
taskNum = sys.argv[2]
taskNum = int(taskNum)
server = tf.train.Server(cluster,job_name=jobType,task_index=taskNum)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding= 'SAME')

def convNN(x):
