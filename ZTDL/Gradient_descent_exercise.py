import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//wines.csv')

y = df['Class']

y_cat = pd.get_dummies(y)

X = df.drop('Class', axis = 1)
 
import seaborn as sns

sns.pairplot(df, hue = 'Class')

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
 
Xsc = sc.fit_transform(X)

from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from keras.layers import Dense
import keras.backend as K

K.clear_session()
model = Sequential()
model.add(Dense(5, input_shape = (13, ), kernel_initializer = 'he_normal',
                activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(RMSprop(lr = 0.1), 'categorical_crossentropy',
              metrics = ['accuracy'])

#By setting verbose 0, 1 or 2 you just say
#how do you want to 'see' the training progress for each epoch.
model.fit(Xsc, y_cat.values, 
          batch_size = 8, 
          epochs = 10,
          verbose = 1, 
          validation_split = 0.2)

K.clear_session()
model = Sequential()
model.add(Dense(8, input_shape = (13, ), 
                kernel_initializer = 'he_normal', activation = 'tanh'))
model.add(Dense(5, kernel_initializer = 'he_normal', activation = 'tanh'))
model.add(Dense(2, kernel_initializer = 'he_normal', activation = 'tanh'))
model.add(Dense(3, kernel_initializer = 'he_normal', activation = 'tanh'))

model.compile(RMSprop(lr = 0.05), 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(Xsc, y_cat.values, 
          batch_size = 16,
          epochs = 20,
          verbose = 1)

print(model.summary())
inp = model.layers[0].input
out = model.layers[2].output

features_function = K.function([inp], [out])
features = features_function([Xsc])[0]
print(features.shape)
#plt.scatter(features[:, 0], features[:, 1], c = y_cat)

from keras.layers import Input
from keras.models import Model

K.clear_session()

#Defining an input layer
inputs = Input(shape = (13, ))

#Defining two hidden layers, one with 8 nodes and other one with 5 nodes
x = Dense(8, kernel_initializer = 'he_normal', activation = 'tanh')(inputs)
x = Dense(5, kernel_initializer = 'he_normal', activation = 'tanh')(x)

#Defining a second_to_last layer with 2 nodes
second_to_last = Dense(2, activation = 'tanh',
                       kernel_initializer = 'he_normal')(x)

#Defining an output layer
outputs = Dense(3, activation = 'softmax')(second_to_last)

#Creating a model that connects input and output
model = Model(inputs = inputs, outputs = outputs)

model.compile(RMSprop(lr = 0.5), 'categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(Xsc, y_cat.values, batch_size = 16, epochs = 20, verbose = 1)

#Defining a function between inpts and second_to_last
features_function = K.function([inputs], [second_to_last])

features = features_function([Xsc])[0]

#plt.scatter(features[:, 0], features[:, 1], c = y_cat)

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

checkpointer = ModelCheckpoint(filepath = 'C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python programmes/Tensor Flow/ZTDL/weights.hdf5',
                               save_best_only = True, verbose = 1) 
earlystopper = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 1, verbose = 1, mode = 'auto')

tensorboard = TensorBoard(log_dir = 'C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/Python programmes/Tensor Flow/ZTDL/tensorboard')

X_train, X_test, y_train, y_test = train_test_split(Xsc, y_cat.values,
                                                    test_size = 0.3,
                                                    random_state = 42)

K.clear_session()

#Defining an input layer
inputs = Input(shape = (13, ))

#Defining two hidden layers, one with 8 nodes and other one with 5 nodes
x = Dense(8, kernel_initializer = 'he_normal', activation = 'tanh')(inputs)
x = Dense(5, kernel_initializer = 'he_normal', activation = 'tanh')(x)

#Defining a second_to_last layer with 2 nodes
second_to_last = Dense(2, activation = 'tanh',
                       kernel_initializer = 'he_normal')(x)

#Defining an output layer
outputs = Dense(3, activation = 'softmax')(second_to_last)

#Creating a model that connects input and output
model = Model(inputs = inputs, outputs = outputs)

model.compile(RMSprop(lr = 0.05), 'categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 20, 
          verbose = 1, validation_data = (X_test, y_test),
          callbacks = [checkpointer, earlystopper, tensorboard])


