import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesData():
  
    def __init__(self, num_points, xmin, xmax):
    
        self.xmax = xmax
        self.xmin = xmin
        self.num_points = num_points
        self.resolution = (xmax-xmin)/(num_points)
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data) # this is the sin function
        
    def ret_true(self, x_series):
        return np.sin(x_series)
    
    def next_batch(self, batch_size, steps, return_batch_ts = False):
        
        #Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)
        
        #Convert to be on time series, it gives starting point on the time series
        ts_start = rand_start * (self.xmax - self.xmin- (steps * self.resolution))

        #Create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0, steps + 1) * (self.resolution)

        #Create the Y data for the time series x axis from previous step
        y_batch = np.sin(batch_ts)        

        #Formatting the RNN
        if return_batch_ts:
             return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:,1:].reshape(-1, steps, 1), batch_ts
        else:
            #The first one represents the time series, and the next one represent the shifted time series by one unit
             return y_batch[:,:-1].reshape(-1, steps, 1), y_batch[:,1:].reshape(-1, steps, 1)
        
#We want 250 points between 0 and 10        
ts_data = TimeSeriesData(250, 0, 10)
#We are just plotting the various x values between 0 to 10 and their corresponding y values in the sin function
'''
plt.plot(ts_data.x_data, ts_data.y_true)
'''
num_time_steps = 30
y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)
'''
#this statement has error because the dimenstion of ts and y2 are not equal because y2 is shiften by unit
plt.plot(ts.flatten(), y2.flatten(), '*')
the dimension of y2 is one less than the dimension of ts. The graph will always be different because we are using random values
#plt.plot(ts.flatten()[1:], y2.flatten(), '*')
'''
'''
#now we are just combining the above two plot statements in one figure
plt.plot(ts_data.x_data, ts_data.y_true, label = 'Sin(t)')
plt.plot(ts.flatten()[1:], y2.flatten(), '--', label = 'Single Training Instance')

plt.legend()
plt.tight_layout()
'''
#TRAINING DATA
train_inst = np.linspace(5, 5 + ts_data.resolution*(num_time_steps+1), num_time_steps + 1)

'''
plt.title("A Training Instance")
#Here we are having our time series
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'bo', markersize = 15, alpha = 0.5, label="Instance")

#here we are generating our new time series
plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), 'ko', markersize = 7, label = 'target')

plt.legend()
'''

#Creating the model 
tf.reset_default_graph()
num_input = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = 2000
batch_size = 1

#Placeholders
X = tf.placeholder(tf.float32, [None, num_time_steps, num_input])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# RNN CELL LAYER 
#we can use different cell i.e. GRUCell and other types of cell
cell = tf.contrib.rnn.BasicRNNCell(num_units = num_neurons, activation = tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
#Mean Square Error
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

#SESSION
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_train_iterations):
        X_batch ,y_batch = ts_data.next_batch(batch_size, num_time_steps)
        
        sess.run(train, feed_dict = {X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict = {X: X_batch, y:y_batch})
            print(iteration, '\tMSE', mse)
            
    saver.save(sess, './rnn_time_series_model_codealong')
"""    
with tf.Session() as sess:
    saver.restore(sess, './rnn_time_series_model_codealong')
    
    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_input)))
    y_pred = sess.run(outputs, feed_dict = {X : X_new})
"""
'''    
plt.title("Testing the model")

#Training the instance
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), 'bo', markersize = 15, alpha = 0.5, label = "Training test")

#Target to predict (correct test values np.sin(train))
plt.plot(train_inst[1:], np.sin(train_inst[1:]), 'ko', markersize = 10, label = 'target')

#Model Prediction    
plt.plot(train_inst[1:], y_pred[0, :, 0], 'r.', markersize = 10, label = 'predictions')

plt.xlabel("time")
plt.legend()
plt.tight_layout()
'''
#Generating a new sequence
with tf.Session() as sess:
    
    saver.restore(sess, './rnn_time_series_model_codealong')
    
    #seed zeros
    
    training_instance= list(ts_data.y_true[:30])
    for iteration  in range(len(ts_data.x_data) - num_time_steps):
    
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        
        y_pred = sess.run(outputs, feed_dict = {X : X_batch})
        
        training_instance.append(y_pred[0, -1, 0])
        
plt.plot(ts_data.x_data, training_instance, 'b-')
plt.plot(ts_data.x_data[:num_time_steps], training_instance[:num_time_steps], 'r', linewidth = 3)
plt.xlabel('TIME')
plt.ylabel('Y')
        