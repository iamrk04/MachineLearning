# Feature Scaling is a must in Deep Learning (has to be applied on entire feature even if a column is already in a good range)
# Please unzip dataset.zip first from https://drive.google.com/file/d/1jyTU7XFESDlx8pqus5xsF1iBrSLUetMa/view?usp=sharing

import numpy as np
import pandas as pd
import tensorflow as tf

# 1 Data Preprocessing
# 1.1 Importing the dataset
dataset = pd.read_excel('7.DeepLearning/Folds5x2_pp.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 1.2 Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# 2 Building the ANN
# 2.1 Initializing the ANN
ann = tf.keras.models.Sequential()

# 2.2 Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# 2.3 Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# 2.4 Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))


# 3 Training the ANN
# 3.1 Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 3.2 Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# 4. Predicting the results of the Test set
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# 5. Evaluating the model performance
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("R^2 Score \t=>", score)