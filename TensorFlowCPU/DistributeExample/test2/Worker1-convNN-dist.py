import tensorflow as tf
import numpy as np
import pickle

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

#distribute
cluster = tf.train.ClusterSpec({'ps':['10.10.1.142:2222'],'worker':['10.10.1.140:2222','10.10.1.141:2222']})
workerStr = '/job:worker/task:'
ps1 = '/job:ps/task:0'
jobType = 'worker'
taskNum = 0
server = tf.train.Server(cluster,job_name=jobType,task_index=taskNum) #worker1


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding= 'SAME')

def convNN(x):

    if jobType == 'ps':
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(worker_device=workerStr+str(taskNum),cluster=cluster)):

            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
                       'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                       'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
                       'out':tf.Variable(tf.random_normal([1024,n_classes]))}

            biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
                       'b_conv2': tf.Variable(tf.random_normal([64])),
                       'b_fc': tf.Variable(tf.random_normal([1024])),
                       'out': tf.Variable(tf.random_normal([n_classes]))}

            #1 layer
            conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
            conv1 = maxpool2d(conv1)

            #2 layer
            conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
            conv2 = maxpool2d(conv2)

            fc = tf.reshape(conv2,shape=[-1,7*7*64])
            fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])

            #dropout will help with large data, helps with weights
            #fc = tf.nn.dropout(fc,keep_rate) #80% of neurons kept

            output = tf.matmul(fc,weights['out'])+biases['out']

            return output

def train_neural_network(x, hm_epochs=1):
    predicition = convNN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicition,labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session(server.target) as ses:
        ses.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = ses.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss+=c
            print('epoch ',epoch+1, 'completed out of', hm_epochs, ' loss:',epoch_loss)

        correct = tf.equal(tf.arg_max(predicition,1),tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)