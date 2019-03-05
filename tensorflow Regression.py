#Tensorflor Regression

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# y = mx+b
#b = 5 taking m and b arbitrary

y_true = (0.5 * x_data)+ 5 + noise

#Creating the data frame
x_df = pd.DataFrame(data = x_data, columns = ['X Data'])
y_df = pd.DataFrame(data = y_true, columns = ['Y'])
my_data = pd.concat([x_df, y_df], axis = 1) # axis 1 means along the columns

#taking a sample from my_data of lenght 50
#this is the initial plot where we are taking the sample and plotting it
my_data.sample(n=250).plot(kind = 'scatter', x = "X Data", y = "Y")

# At a single time we cannot feed a large amount of data once in the neural network so we create batches to do that
# we feed the data in different amount of batches that are efficient

batch_size = 8 # 8 is just arbitrary 
m = tf.Variable(0.81)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m*xph + b
error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    batches = 10000
    
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size = batch_size)
        feed = {xph: x_data[rand_ind], yph:y_true[rand_ind]}
        sess.run(train, feed_dict = feed)
        
    model_m, model_b = sess.run([m, b])
    
print(model_m, model_b)

y_hat = x_data*model_m + model_b
my_data.sample(250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.plot(x_data, y_hat, 'r')
plt.show()
    
#Now we are learning tf estimator
#here we are creating the feature column list for the estimator
feat_cols = [ tf.feature_column.numeric_column('x', shape = [1]) ]
estimator = tf.estimator.LinearRegressor(feature_columns = feat_cols)

from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size = 0.3, random_state = 101) 

input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size = 8, num_epochs = None, shuffle= True)

#this is the same as above except that we will change num_epochs to 1000 and shuffle to False
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size = 8, num_epochs = 1000, shuffle= False)

eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size = 8, num_epochs = 1000, shuffle= False)

estimator.train(input_fn = input_func, steps = 1000)

train_metrics = estimator.evaluate(input_fn = train_input_func , steps = 1000)

eval_metrics = estimator.evaluate(input_fn = eval_input_func, steps = 1000)

print("Training Data Metrics")
print(train_metrics)

print("Eval Metrics")
print(eval_metrics)

# A good indication that we are going good is that the loss in eval_metrics is greater than the loss in train_metric loss

#To predict new values

brand_new_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle = False)

#its a generator object, so to see the result we need to iterate or use list
print(list(estimator.predict(input_fn = input_fn_predict)))

predictions = []

for pred in estimator.predict(input_fn = input_fn_predict):
    predictions.append(pred['predictions'])

print(predictions)

my_data.sample(n= 250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.plot(brand_new_data, predictions, 'r*')