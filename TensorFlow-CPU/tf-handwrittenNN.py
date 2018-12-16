import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#10 classes, 0-9

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# 784 values (pixels)
x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):
    # input data * weights + bias

    hiddenlayer1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                    'biases': tf.Variable(tf.random_normal(n_nodes_hl1))}
    hiddenlayer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'biases': tf.Variable(tf.random_normal(n_nodes_hl2))}
    hiddenlayer3= {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases': tf.Variable(tf.random_normal(n_nodes_hl3))}
    outputlayer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal(n_classes))}

    L1 = tf.add(tf.matmul(data,hiddenlayer1['weights']) + hiddenlayer1['biases'])
    L1 = tf.nn.relu(L1)

    L2 = tf.add(tf.matmul(L1, hiddenlayer2['weights']) + hiddenlayer2['biases'])
    L2 = tf.nn.relu(L2)

    L3 = tf.add(tf.matmul(L2, hiddenlayer3['weights']) + hiddenlayer3['biases'])
    L3 = tf.nn.relu(L3)

    Loutput = tf.matmul(L3, outputlayer['weights']) + outputlayer['biases']

    return Loutput