# This is suitable for high dimensional datasets - which has lots of features (independent variables)
# Built on top of Decision Tree
# Feature scaling should never be applied to Decision Tree Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("1.Regression/Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualising the Random Forest Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Evaluating the model performance
# from sklearn.metrics import r2_score
# score = r2_score(y_test, y_pred)
# print("R^2 Score \t=>", score)