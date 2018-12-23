import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import datetime
import sys

lemmatizer = WordNetLemmatizer()

nclasses = 2 # 0=neg, 4=pos # 1 or 0 now

batch_size = 32
total_batches = 128

n_nodes_hl1 = 500
n_nodes_hl2 = 500

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

with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
    counter = 0
    for line in f:
        counter+=1
    print("lines: "+counter)

def neural_network_model(data):
    if jobType == 'ps':
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(worker_device=workerStr+str(taskNum),cluster=cluster)):
            hidden_1_layer = {'f_fum': n_nodes_hl1,
                              'weight': tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                              # 2638 is the number of words in the lexicon
                              'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

            hidden_2_layer = {'f_fum': n_nodes_hl2,
                              'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                              'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

            output_layer = {'f_fum': None,
                            'weight': tf.Variable(tf.random_normal([n_nodes_hl2, nclasses])),
                            'bias': tf.Variable(tf.random_normal([nclasses]))}

            l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
            l1 = tf.nn.relu(l1)
            l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
            l2 = tf.nn.relu(l2)
            output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
            return output

def estimated_time(count,oldtime):
    time = datetime.datetime.now().time()
    e1 = int(time.second)
    e2 = int(oldtime.second)
    timeDif = e2-e1
    if(timeDif < 1):
        timeDif = 1
    totalLeft = total_batches-count
    estimate = timeDif*totalLeft
    minutes = estimate / 60
    if (minutes > 1):
        estimate = minutes
        w = "M"
        if(estimate > 60):
            estimate = estimate/60
            w = "H"
    else:
        w = 'S'
    return time, estimate, w

def train_neural_network(x, hmEpochs=1):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session(server.target) as sess:
        for epoch in hmEpochs:
            epoch_loss = 1
            with open('lexicon.pickle', 'rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                oldtime = datetime.datetime.now()
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))
                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                      y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        oldtime, remaining, w = estimated_time(batches_run, oldtime)
                        print('Batch run:', batches_run, '/', total_batches, '|', w, 'Remaining:', '%4.2f' % remaining,
                              '| Epoch:', epoch+1, '| Batch Loss:', c, )


train_neural_network(x)