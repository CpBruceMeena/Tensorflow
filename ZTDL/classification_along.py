import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//user_visiting_duration.csv')

print(df.head())

df.plot(kind = 'scatter', x = 'Time (min)', y = 'Buy')

from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(1, input_shape = (1, ), activation = 'sigmoid'))

#here we are using binary crossentropy instead of mean squared error
model.compile(SGD(lr = 0.5), 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

X = df['Time (min)'].values
y = df['Buy'].values

model.fit(X, y, epochs = 25)
ax = df.plot(kind = 'scatter', x = 'Time (min)', y = 'Buy', 
             title = 'Purchase behaviour VS time spent on site')

temp = np.linspace(0, 4)
ax.plot(temp, model.predict(temp), color = 'orange')
plt.legend(['model', 'data'])

temp_class = model.predict(temp)>0.5
ax = df.plot(kind = 'scatter', x = 'Time (min)', y = 'Buy',
             title = 'Purchase behaviour vs time spent on site')

temp = np.linspace(0, 4)
ax.plot(temp, temp_class , color = 'orange')
plt.legend(['model', 'data'])

y_pred = model.predict(X)
y_class_pred = y_pred > 0.5 

from sklearn.metrics import accuracy_score
print('the accuracy score is {:0.3f}'.format(accuracy_score(y, y_class_pred)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

params = model.get_weights()
params = [np.zeros(w.shape) for w in params]
model.set_weights(params)

print('the accuracy score is {:0.3f}'.format(accuracy_score(y, model.predict(X)>0.5)))

model.fit(X_train, y_train, epochs = 30, verbose = 0)

print('the train accuracy score is {:0.3f}'.format(accuracy_score(y_train, model.predict(X_train) > 0.5)))
print('the test accuracy score is {:0.3f}'.format(accuracy_score(y_test, model.predict(X_test) > 0.5))) 