# Simple Linear Regression in Python
# Tutorial via Weird Geek - weirdgeek.com/2018/11/linear-regression-model-using-python/
# data set: https://www.weirdgeek.com/wp-content/uploads/2018/11/Boston_housing.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

# introductory analysis of data set
data = pd.read_csv("../input/boston-housingcsv/Boston_housing.csv")
data.head()
data.info()
data.columns
data.describe()

# data reshaping
x = data.drop("medv", axis = 1).values
y = data["medv"].values
X_rooms = x[:, 5]
type(X_rooms), type(y)
X_rooms = X_rooms.reshape(-1,1)
y_train = y.reshape(-1,1)
print(X_rooms.shape, y_train.shape)

# heatmap intial analysis
sns.heatmap(data.corr(), square=True, cmap='RdYlGn')

#scatter plot analysis
plt.scatter(X_rooms, y_train, color='green', s=3)
plt.ylabel("Value of house/1000($)")
plt.xlabel("Number of rooms")
plt.show()

# full regression
regression = linear_model.LinearRegression()
regression.fit(X_rooms, y_train)
plt.scatter(X_rooms, y, color='green', s=3)
#We want to check out the regressor predictions over the range of data, we can do so:
Data_range = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)
y_pred = regression.predict(Data_range)
plt.plot(Data_range, y_pred , color="black", linewidth = 3)
plt.show()
