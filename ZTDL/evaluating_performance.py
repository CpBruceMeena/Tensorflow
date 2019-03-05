import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//height_weight.csv')

X = df["Height"].values
y_true = df['Weight'].values


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

model = Sequential()
model.add(Dense (1, input_shape= (1, )))
#the below command is to read the summary of the model basically the components and other stuff
model.summary()

model.compile(Adam(lr = 0.8), 'mean_squared_error')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size = 0.2)

model.fit(X_train, y_train, epochs = 50, verbose = 0)
y_train_pred = model.predict(X_train).ravel()
y_test_pred = model.predict(X_test).ravel()
 
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

print("the mean squared error on the train set is :\t{:0.1f}".format(mse(y_train, y_train_pred)))
print("the mean squared error on the test set is :\t{:0.1f}".format(mse(y_test, y_test_pred)))

print("the r2 score on the train set is :\t{:0.1f}".format(r2_score(y_train, y_train_pred)))
print("the r2 score on the test set is :\t{:0.1f}".format(r2_score(y_test, y_test_pred)))

W, B = model.get_weights()
print(W, B)