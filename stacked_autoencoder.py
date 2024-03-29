import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//Python programmes//Tensor Flow//MNIST_data', one_hot = True)

tf.reset_default_graph()

#784 inputs (28 * 28 pixels)
#392 half of the above
#196 again halving
#392
#784

num_inputs = 784
neurons_hid1 = 392
neurons_hid2 = 196
neurons_hid3 = neurons_hid1
num_outputs = num_inputs
 
learning_rate = 0.01
actf = tf.nn.relu
X = tf.placeholder(tf.float32, shape = [None, num_inputs])
initializer = tf.variance_scaling_initializer()

w1 = tf.Variable(initializer([num_inputs , neurons_hid1]), dtype = tf.float32)
w2 = tf.Variable(initializer([neurons_hid1 , neurons_hid2]), dtype = tf.float32)
w3 = tf.Variable(initializer([neurons_hid2 , neurons_hid3]), dtype = tf.float32)
w4 = tf.Variable(initializer([neurons_hid3 , num_outputs]), dtype = tf.float32)


b1 = tf.Variable(tf.zeros(neurons_hid1))
b2 = tf.Variable(tf.zeros(neurons_hid2))
b3 = tf.Variable(tf.zeros(neurons_hid3))
b4 = tf.Variable(tf.zeros(num_outputs))

act_func = tf.nn.relu

hid_layer1 = act_func(tf.matmul(X, w1)+b1)
hid_layer2 = act_func(tf.matmul(hid_layer1, w2)+b2)
hid_layer3 = act_func(tf.matmul(hid_layer2, w3)+b3)
output_layer = act_func(tf.matmul(hid_layer3, w4)+b4)

loss = tf.reduce_mean(tf.square(output_layer - X))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

num_epochs = 5
batch_size = 150

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(num_epochs):
        
        num_batches = mnist.train.num_examples//batch_size
        
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict = {X:X_batch})
            
        training_loss = loss.eval(feed_dict = {X:X_batch})
        print("EPOCH: {} LOSS: {}".format(epoch, training_loss))
    
    saver.save(sess, './example_stacked_autoencoder.ckpt')
    
num_test_images = 10

with tf.Session() as sess:
    
    saver.restore(sess, './example_stacked_autoencoder.ckpt')
    
    results = output_layer.eval(feed_dict = {X:mnist.test.images[:num_test_images]})

f, a = plt.subplots(2, 10, figsize=(20, 4))

for i in range(num_test_images):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(results[i], (28, 28)))
    
    
        
        