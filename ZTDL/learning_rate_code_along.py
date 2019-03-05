import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//bank_notes.csv')

import seaborn as sns

sns.pairplot(df, hue = 'class')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale

X = scale(df.drop('class', axis = 1).values)
y = df['class'].values
 
model = RandomForestClassifier()
cross_val_score(model, X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)

from keras.models import Sequential
from keras.layers import Dense, Activation 
from keras.optimizers import SGD

import keras.backend as K

K.clear_session()

model= Sequential()
model.add(Dense(1, input_shape = (4, ), activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd',
              metrics = ['accuracy'])

history = model.fit(X_train, y_train)
result = model.evaluate(X_test, y_test)

historydf = pd.DataFrame(history.history, index = history.epoch)
historydf.plot(ylim = (0, 1))

plt.title("Test accuracy: {:3.1f} %".format(result[0]*100), fontsize = 15)

dflist = [] 
learning_rate = [0.05, 0.01, 0.1, 0.5]

for lr in learning_rate:
    K.clear_session()
    model = Sequential()
    model.add(Dense(1, input_shape = (4, ), activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', 
                  optimizer = SGD(lr = lr), 
                  metrics = ['accuracy'])
    
    h = model.fit(X_train, y_train, batch_size = 16, verbose = 0)
    dflist.append(pd.DataFrame(h.history, index = h.epoch))
    
historydf = pd.concat(dflist, axis = 1)
metrics_reported = dflist[0].columns 
idx = pd.MultiIndex.from_product([learning_rate, metrics_reported],
                                  names = ['learning_rate', 'metric'])

historydf.columns = idx
ax = plt.subplot(211)
historydf.xs('loss', axis = 1, level = 'metric').plot(ylim = ( 0, 1), ax = ax)
plt.title("Loss")

ax = plt.subplot(212)
historydf.xs('acc', axis = 1, level = 'metric').plot(ylim = (0, 1), ax = ax)
plt.title("Accuracy")
plt.xlabel('Epochs')

plt.tight_layout()

