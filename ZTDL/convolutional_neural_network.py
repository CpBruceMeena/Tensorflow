import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.datasets import mnist 

(X_train, y_train), (X_test, y_test) = mnist.load_data('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//Python programmes//Tensor Flow//ZTDL//mnist.npz')

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.0
X_test /= 255.0

from keras.utils.np_utils import to_categorical

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

K.clear_session()
model = Sequential()
model.add(Dense(512, input_dim = 28*28, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop',
              metrics = ['accuracy'])

h = model.fit(X_train, y_train_cat, batch_size = 128, epochs = 10, verbose = 1, validation_split = 0.2)

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Accuracy'])
plt.title('Accuracy')
plt.xlabel('Epochs')

test_accuracy = model.evaluate(X_test, y_test_cat)[1]
