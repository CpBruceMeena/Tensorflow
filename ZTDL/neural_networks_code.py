import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons 

X, y = make_moons(n_samples = 1000, noise = 0.1, random_state = 0)

plt.plot(X[y == 0, 0], X[y == 0,1], 'ob', alpha = 0.5)
plt.plot(X[y == 1, 0], X[y == 1,1], 'xr', alpha = 0.5)

plt.legend(['0', '1'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense

#Shallow model 
 
model = Sequential()
model.add(Dense(1, input_shape = (2, ), activation = 'sigmoid'))
model.compile(SGD (lr = 0.05), 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 200, verbose = 0)

results = model.evaluate(X_test, y_test)

print('The accuracy score on the train set is :\t{:0.3f}'.format(results[1]))

def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis = 0) - 0.1
    amax, bmax = X.max(axis = 0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    c = model.predict(ab)
    cc = c.reshape(aa.shape)
    
    plt.figure(figsize = (12, 8))
    plt.contourf(aa, bb, cc, cmap = 'bwr', alpha = 0.2)
    plt.plot(X[y == 0,0], X[y == 0, 1], 'ob', alpha = 0.5)
    plt.plot(X[y == 1,0], X[y == 1, 1], 'xr', alpha = 0.5)
    
    plt.legend(['0', '1'])

plot_decision_boundary(model, X, y) 

#Deep model

model = Sequential()
model.add(Dense(4, input_shape = (2, ), activation = 'tanh'))
model.add(Dense(2, activation = 'tanh'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(SGD(lr = 0.05), 'binary_crossentropy', metrics = ['accuracy'])


model.fit(X_train, y_train, epochs = 100, verbose = 0)
  
model.evaluate(X_test, y_test)

from sklearn.metrics import accuracy_score, confusion_matrix

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)

print('HELLO')
print("The accuracy score on the train set is :\t{:0.3f}".format(accuracy_score(y_train, y_train_pred)))
print("The accuracy score on the test set is :\t{:0.3f}".format(accuracy_score(y_test, y_test_pred)))

plot_decision_boundary(model, X, y)

 