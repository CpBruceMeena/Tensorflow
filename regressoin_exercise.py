#We are trying the regression for the median housing price

import tensorflow as tf
import pandas as pd

housing = pd.read_csv('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/cal_housing.csv', error_bad_lines = False)

#First we are normalizing the columns 
cols_to_norm = ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population',
       'households', 'medianincome', 'medianHouseValue']

housing[cols_to_norm] = housing[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))

y_val = housing['medianHouseValue']
x_data = housing.drop('medianHouseValue', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size = 0.3, random_state = 101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(data = scaler.transform(X_train), columns = X_train.columns, index = X_train.index)
X_test = pd.DataFrame(data = scaler.transform(X_test), columns = X_test.columns, index = X_test.index)

#We are taking the featuer column and then taking then to numeric_column
housing_med = tf.feature_column.numeric_column('housingMedianAge')
total_room = tf.feature_column.numeric_column('totalRooms')
total_bedroom = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
household = tf.feature_column.numeric_column('households')
median_income = tf.feature_column.numeric_column('medianincome')

feat_cols = [housing_med, total_room, total_bedroom, population,household, median_income]

input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 10, num_epochs = 1000, shuffle = True)

model = tf.estimator.DNNRegressor(hidden_units = [6, 6, 6], feature_columns = feat_cols)
model.train(input_fn = input_func, steps = 2000)

predict_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = 10, num_epochs=1, shuffle = False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)

print(predictions)

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, final_preds)**0.5
print(mean_squared_error)
  
    