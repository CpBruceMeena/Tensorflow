import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//Python programmes//Tensor Flow//MNIST_data', one_hot = True)

# We are creating helper functions

#Init weights 
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init_random_dist)

#Init bias
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias_vals)
    
#Conv2D
def conv2d(x, W):
    # x --> [batch, H, W, channels]
    # W --> [filter H, filter W, channels IN, channesl OUT]
    return tf.nn.conv2d(x,W, strides = [1, 1, 1, 1], padding = 'SAME')

#Pooling
def max_pool_2by2(x):
    # x --> [batch, h, w, c]
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1, 2, 2, 1], padding ='SAME')
    
# CONVOLUTIONAL LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

#NORMAL (FULLY CONNECTED)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W)+b

# PLACEHOLDER 
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape = [None, 10])

#Layer
#1 is for grayscale
#28, 28 is the height and widht
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 32 are the number of features and 5,5 patch size, 1 is the input channel i.e. grayscale
convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
convo_1_pooling = max_pool_2by2(convo_1)
convo_2 = convolutional_layer(convo_1_pooling, shape = [5, 5, 32, 64])

convo_2_pooling = max_pool_2by2(convo_2)
convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))
  
#Dropout
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob = hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10)

#LOSS Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
steps = 5000
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        
        sess.run(train, feed_dict = {x: batch_x, y_true: batch_y, hold_prob : 0.5})
        
        if i%100 == 0:
            print('ON STEP: {}'.format(i))
            print('ACCURACY: ')
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(sess.run(acc, feed_dict = {x: mnist.test.images, y_true: mnist.test.labels, hold_prob: 1.0}))
            print('\n')

        
