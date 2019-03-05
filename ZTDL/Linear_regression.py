import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('C://Users//LENOVO IDEAPAD 320//OneDrive//Desktop//height_weight.csv')
print(df.head())

df.plot(kind = 'scatter', 
        x = 'Height',
        y = 'Weight',
        title = "Initial Weight and Height")

plt.plot([55, 78], [75, 250], color = 'red', linewidth = 3)

def line(x, w=0, b=0):
    return x*w + b

x = np.linspace(55, 80, 100)
yhat = line(x, w = 0, b = 0)
print(yhat)

df.plot(kind = 'scatter',
        x = "Height",
        y = "Weight", 
        title = "Weight and Height")
 
plt.plot(x, yhat, color = 'red', linewidth = 3)

def mean_squared_error(y_true, y_pred):
    s = (y_pred - y_true)**2
    return s.mean()

x = df[["Height"]].values
y_true = df['Weight'].values

y_pred = line(x)
print(mean_squared_error(y_true, y_pred.ravel()))

