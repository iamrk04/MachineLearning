# Feature Scaling is a must in Deep Learning (has to be applied on entire feature even if a column is already in a good range)
# Please unzip dataset.zip first from https://drive.google.com/file/d/1jyTU7XFESDlx8pqus5xsF1iBrSLUetMa/view?usp=sharing

import numpy as np
import pandas as pd
import tensorflow as tf

# 1 Data Preprocessing
# 1.1 Importing the dataset
dataset = pd.read_csv("7.DeepLearning/Churn_Modelling.csv")
x = dataset.iloc[:, 3: -1].values
y = dataset.iloc[:, -1].values

# 1.2 Encoding categorical data
# 1.2.1 Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# 1.2.2 One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# 1.3 Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# 1.4 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# 2 Building the ANN
# 2.1 Initializing the ANN
ann = tf.keras.models.Sequential()

# 2.2 Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))      # units have to be tested out with different values to see which is performing better

# 2.3 Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# 2.4 Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))   # activation='softmax' for non-binary classification


# 3 Training the ANN
# 3.1 Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])   # loss='categorical_crossentropy' for non-binary classification

# 3.2 Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)    # batch_size = 32 works well, try to optimize this if you have time


# 4. Making the predictions and evaluating the model
# 4.1 Predicting the result of a single observation
# careful with input for prediction - has to be encoded and then feature scaled
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# 4.2 Predicting the Test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print("Predict \t=>", y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# 4.3 Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix\n", cm)
score = accuracy_score(y_test, y_pred)
print("Score \t=>", score)
