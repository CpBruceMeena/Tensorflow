import pandas as pd

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//user_visiting_duration.csv')

X = df['Time (min)'].values
y = df['Buy'].values


def build_logistic_regression_model():
    model = Sequential()
    model.add(Dense(1, input_shape = (1, ), activation = 'sigmoid'))
    model.compile(SGD( lr = 0.5), 
                  'binary_crossentropy', 
                  metrics = ['accuracy'])
    return model
    
model = KerasClassifier(build_fn = build_logistic_regression_model, epochs = 25,
                        verbose = 0)

from sklearn.model_selection import cross_val_score, KFold

cv = KFold(3, shuffle = True)
scores = cross_val_score(model, X, y, cv = cv)
print(scores)
print('the validation accuracy is {:0.4f}-+{:0.4f}'.format(scores.mean(), scores.std()))
