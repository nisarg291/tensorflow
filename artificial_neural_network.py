# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# dependent varables are from creditscore to EstimatedSalary columns are dependent variables
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
# in label encoding the columns name is same as Gender but value is become 0 or 1, 0 for female and 1 for male  
from sklearn.preprocessing import LabelEncoder
# le1 = LabelEncoder()
# X[:, 1] = le1.fit_transform(X[:, 1])
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Building the ANN

# Initializing the ANN
# in we initialized ann with sequential(sequence) of layers
# or we can define ann with graph
# tf.keras.models.Sequential() it for define ann with sequence of layers
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# outputdim is number of node 
# rectified fun is use when we give the value relu to the activation para.
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
# sigmoid fun is useful for output
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
# if dependent variable have value 0 and 1 so here it is loss=binary_crossentropy
# if dependent variable have more then 2 outcomes then loss=categorcal_croosentropy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
# it is for when y_pred is more than 0.5(equal to 1 bcz ans 0 kato 1 ma hoy) that means it is in more risk that that customer can leave the bank 
y_pred = (y_pred > 0.5)
# so now value of y_pred is true if the value of y_pred >0.5 and false if value of y_pred is <0.5
print(y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)