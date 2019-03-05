import pandas as pd
import numpy as np

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//iris.csv')

import seaborn as sns

sns.pairplot(df, hue = 'species')
print(df.head())

X = df.drop('species', axis = 1)
print(X.head())

target_names = df['species'].unique()
print(target_names)

target_dict = {n:i for i,n in enumerate(target_names)}
print(target_dict)
 
y = df['species'].map(target_dict)

from keras.utils.np_utils import to_categorical

#We are using to_categorical to convert the results into matrix from where we'll have zeros for the non existent part
# and 1 where the species exist
y_cat = to_categorical(y)
print(y_cat)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat, test_size = 0.2)

model = Sequential()
model.add(Dense(3, input_shape = (4, ), activation = 'softmax'))
model.compile(Adam(lr = 0.1),loss = 'categorical_crossentropy', metrics = ['accuracy'])

#validation split takes out 10 percent of the training data for checking the accuracy and calculating the loss
model.fit(X_train, y_train, epochs = 20, validation_split = 0.1)

#Now we are calculating the probability in which the element belongs
y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test, axis = 1)
y_pred_class = np.argmax(y_pred, axis = 1)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_class, y_pred_class))

print(confusion_matrix(y_test_class, y_pred_class))
