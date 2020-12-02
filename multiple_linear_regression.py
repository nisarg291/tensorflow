# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# labelencoder_X=LabelEncoder()
# X[:,3]=labelencoder_X.fit_transform(X[:,3])
# onehotencoder=OneHotEncoder(categories=[3])
# X=onehotencoder.fit_transform(X).toarray()
# print(X)#
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
plt.plot(range(0,len(y_test)),y_test,'o')
plt.plot(range(0,len(y_test)),y_pred,'o')
plt.grid()
plt.show()

#Building the optimal model using backword elimination

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,3,4,5]]
reg_OLS=sm.OLS(y,X_opt)
reg_OLS.summary()
X_opt=X[:,[0,3]]
reg_OLS=sm.OLS(y,X_opt)
reg_OLS.summary()






