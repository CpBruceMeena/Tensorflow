import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits = load_digits()

X, y = digits.data, digits.target
'''
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(X.reshape(-1, 8, 8)[i], cmap = 'gray')
   ''' 
from sklearn.model_selection import learning_curve
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K

K.clear_session()

model = Sequential()
#16 nodes and number of output is 64
model.add(Dense(16, input_shape = (64, ), activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile('adam', 'categorical_crossentropy', metrics = ['accuracy'])

initial_weights = model.get_weights()
y_cat = to_categorical(y, 10)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size = 0.3)

train_sizes = (len(X_train)*np.linspace(0.1, 0.999, 4)).astype(int)
train_scores = []
test_scores = []
for train_size in train_sizes:
    #the blank spaces below are for the test samples of both X, and Y which we are not going to use right now
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train, y_train, test_size= train_size)
    
    model.set_weights(initial_weights)
    h = model.fit(X_train_frac, y_train_frac, verbose = 0, epochs = 300,
                  callbacks = [EarlyStopping(monitor = 'loss', patience = 1)])
    
    r = model.evaluate(X_train_frac, y_train_frac, verbose = 0)
    train_scores.append(r[-1])
    
    e = model.evaluate(X_test, y_test, verbose = 0)
    test_scores.append(e[-1])
    
    print('done size:' , train_size)

plt.plot(train_sizes, train_scores, 'o-', label = 'Training score')
plt.plot(train_sizes, test_scores, 'o-', label = ' Test score')
plt.legend(loc = 'best')
    
    
