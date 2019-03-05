import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

milk = pd.read_csv('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/monthly-milk-production-pounds-p.csv', index_col = 'Month')
print(milk.head())
# milk.index = pd.to_datetime(milk.index)
#milk.plot()

#print(milk.info())
train_set = milk.head(156)
test_set = milk.tail(12)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

def next_batch(training_data, batch_size, steps):
    
    rand_start = np.random.randint(0, len(training_data)-steps)
    
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(1, steps+1)
    
    return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

num_inputs = 1
num_time_steps = 12
num_neurons = 100
num_outputs = 1
learning_rate = 0.001
num_iterations = 6000
batch_size = 1

#the reason we are using num_time_steps-1 is because each of these will be one steps shorted than the original time steps size
X = tf.placeholder(tf.float32, shape = [None, num_time_steps-1, num_inputs])
y = tf.placeholder(tf.float32, shape = [None, num_time_steps-1, num_outputs])

cell = tf.contrib.rnn.GRUCell(num_units = num_neurons, activation = tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

#LOSS Function
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss) 
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_iterations):
        
        X_batch, y_batch = next_batch(train_scaled, batch_size, num_time_steps)
        
        sess.run(train, feed_dict = { X : X_batch, y: y_batch})
        
        if iteration %100 == 0:
            mse = loss.eval(feed_dict = {X : X_batch, y : y_batch})
            print(iteration, '\tMSE', mse) 
    
    saver.save(sess,'./ex_time_series_model_codealong')

#Generative session
with tf.Session() as sess:
    saver.restore(sess,'./ex_time_series_model_codealong')
    train_seed = list(train_scaled[-12:])
    
    for iteration in range(12):
        x_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        
        y_pred = sess.run(outputs, feed_dict = {X: x_batch})
        train_seed.append(y_pred[0, -1, 0])
print(train_seed)
        
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12, 1))
        
test_set['Generated'] = results

test_set.plot()

