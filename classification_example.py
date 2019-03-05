import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

diabetes = pd.read_csv('C:/Users/LENOVO IDEAPAD 320/OneDrive/Desktop/pima-indians-diabetes.csv', error_bad_lines = False)

print(diabetes.columns)

cols_to_norm = ['Number_pregnant', 'Glucose_Concentration', 'Blood_pressure', 'Triceps',
       'Insulin ', 'BMI', 'Pedigree', 'Age', 'Class']

#Here we are normalizing the data
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/(x.max()-x.min()))

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_Concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Groups', ['A', 'B', 'C', 'D'])

#In case we dont know the groups
#assigned_group = tf.feature_column.categorical_column_with_hash_bucket("Group", hash_bucket_size = 10)

diabetes['Age'].hist(bins = 20)
plt.show()

#this create buckets of the size that we have written
age_bucket = tf.feature_column.bucketized_column(age, boundaries = [20, 30, 40, 50, 60, 70, 80, 90])

feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_bucket]

# Trains test split
x_data = diabetes.drop('Class', axis = 1)

labels = diabetes['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size = 0.3, random_state = 101)

#this is for the training data
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y = y_train, batch_size = 10,num_epochs = 1000, shuffle = True)

model = tf.estimator.LinearClassifier(feature_columns = feat_cols, n_classes = 2)
model.train(input_fn = input_func, steps = 1000)

#this is for the test data
eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size = 10, num_epochs = 1, shuffle = False)

results = model.evaluate(eval_input_func) 

#This is for the prediction of the input
pred_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = 10, num_epochs =1 ,shuffle = False)
predictions = model.predictions(pred_input_func)
my_pred = list(predictions)

#10, 10 ,10 means three hidden laters with 10 neurons each
dnn_model = tf.estimator.DNNClassifier(hidden_units = [10, 10, 10], feature_columns = feat_cols, n_classes = 2)

dnn_model.train(input_fn = input_func, steps = 1000)
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension = 4)

feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_col, age_bucket]

input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size = 10, num_epochs = 1000, shuffle = True)

dnn_model = tf.estimator.DNNClassifier(hiddnen_units = [10, 10, 10], feature_columns = feat_cols, n_classes = 2)
 
dnn_model.train(input_func, steps = 1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test, batch_size =10, num_epochs = 1, shuffle = False )

dnn_model.evaluate(eval_input_func)






