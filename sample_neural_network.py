
import tensorflow as tf
import numpy as np

"""
Simple Neural Network

n_features = 10
n_dense_neurons = 3

#The rows are the number of samples and the columns are the number of features
x = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))

#b is called the bias term
b = tf.Variable(tf.ones([n_dense_neurons]))

#We are using the tensorflow operations to add and multiply
#z = xW+b
xW = tf.matmul(x, W)
z = tf.add(xW, b)

#Defining the activation function
a = tf.sigmoid(z)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    #Always remember while using feed_dict use : instead of = use :
    layer_out = sess.run(a, feed_dict = {x : np.random.random([1, n_features])})
    print(layer_out)
"""

# Simple Regression Example
import matplotlib.pyplot as plt

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10)+ np.random.uniform(-1.5, 1.5, 10)
plt.plot(x_data, y_label, '*')
plt.show()

#m and b are some random values
m = tf.Variable(.44)
b = tf.Variable(.87)

error = 0
 
# y = mx+b

for x,y in zip(x_data, y_label):
    y_hat = m*x + b
    error += (y-y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    training_steps = 100
    for i in range(training_steps):
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m,b])

x_test = np.linspace(-1, 11, 10)

# y = mx+b

y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()
    
    