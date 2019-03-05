import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//diabetes.csv')

_ = df.hist(figsize = (12, 10))

import seaborn as sns

sns.pairplot(df, hue = 'Outcome')

sns.heatmap(df.corr(), annot = True)

from sklearn.preprocessing import StandardScaler 
from keras.utils.np_utils import to_categorical

sc = StandardScaler()
X = sc.fit_transform(df.drop('Outcome', axis = 1))
y = df['Outcome'].values
y_cat = to_categorical(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, random_state = 22, test_size = 0.2)

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32, input_shape = (8,), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(Adam(lr = 0.05), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 20 , verbose = 2, validation_split = 0.1)
 
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis = 1)
y_test_class = np.argmax(y_test, axis = 1)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
accuracy_score(y_test_class, y_pred_class)

print(classification_report(y_test_class, y_pred_class))
confusion_matrix(y_test_class, y_pred_class)

