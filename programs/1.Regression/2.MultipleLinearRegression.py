import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("1.Regression/50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print("Prediction    Actual \n", np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making a single prediction
print("Predict \t=>", regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Getting the final linear regression equation with the values of the coefficients
print("Coefficent \t=>", regressor.coef_)
print("Intercept \t=>", regressor.intercept_)

# Evaluating the model performance
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("R^2 Score \t=>", score)