import tensorflow as tf
from posneg import create_featureset_and_labels
import numpy as np

train_x,train_y,test_x,test_y = create_featureset_and_labels('/home/ckinney/MachineLearning/Datasets/PosNeg_pt5/pos.txt', '/home/ckinney/MachineLearning//Datasets/PosNeg_pt5/neg.txt')


#10 classes, 0-9

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 50

# 784 values (pixels)
x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):
    # input data * weights + bias

    hiddenlayer1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hiddenlayer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hiddenlayer3= {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    outputlayer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    L1 = tf.add(tf.matmul(data,hiddenlayer1['weights']), hiddenlayer1['biases'])
    L1 = tf.nn.relu(L1)

    L2 = tf.add(tf.matmul(L1, hiddenlayer2['weights']),hiddenlayer2['biases'])
    L2 = tf.nn.relu(L2)

    L3 = tf.add(tf.matmul(L2, hiddenlayer3['weights']), hiddenlayer3['biases'])
    L3 = tf.nn.relu(L3)

    Loutput = tf.matmul(L3, outputlayer['weights']) + outputlayer['biases']

    return Loutput

def train_neural_network(x):
    predicition = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicition,labels=y))


    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 50

    with tf.Session() as ses:
        ses.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = ses.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epoch_loss+=c
                i+=batch_size

            print('epoch ',epoch+1, 'completed out of', hm_epochs, ' loss:',epoch_loss)

        correct = tf.equal(tf.arg_max(predicition,1),tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)