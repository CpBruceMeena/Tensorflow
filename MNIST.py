import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)

mnist.train.images
mnist.train.num_examples # there are total 55000 images
mnist.test.num_examples # there are total 10000 images
import matplotlib.pyplot as plt

#We are reshaping the image to convert it from a single length vector to an array which will give us a image of 
# pixel size 28, 28
single_image = mnist.train.images[1].reshape(28, 28)
print(single_image)

#We are not using colors, instead we'll be using grayscale image
plt.imshow(single_image, cmap = 'gist_gray')
print(single_image.min(), single_image.max())

#we are using 784 i.e is 28*28
x = tf.placeholder(tf.float32, shape=[None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W)+b

#loss function
y_true = tf.placeholder(tf.float32,[None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
 
with tf.Session() as sess:
    
    sess.run(init)
    for step in range(1000):
        batch_x, batch_y =  mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
    
    #Evaluate the model 
    #the next statement will return the index position the y witht the highest probability
    # 1 describes the axis along which to search
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
    print(tf.argmax(y, 1), tf.argmax(y_true, 1))
    #this staetment will return a list of boolean values that is True and False
    # and we want to convert this true and false values to binary values i.e. 0 and 1
    #We are using casting to do the above conversion
    #[True, False, True,...] ---> [1, 0, 1,...]
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Predicted [3, 4] let the first number predicted is 3 and the other one is 4
    #for the true data let the numbers be [3, 9], so we have got the first one correct and the second one wrong
    #[True, False] --> [1.0 , 0.0], reduce_mean will give us the average i.e. is [0.5]
    
    print(sess.run(acc,feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))
     