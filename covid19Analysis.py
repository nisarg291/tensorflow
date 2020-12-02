# # Polynomial Regression

# # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('nation_level_daily1.csv',index_col='index')
# dataset['index']=dataset.index.values
# dataset.to_csv('nation_level_daily1.csv')
X=dataset.iloc[:,0:1].values;
# print(X)
y = dataset.iloc[:, 2].values
# print(y)
# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# [694333.33333333]
# [4190208.38932806]
# Visualising the Linear Regression results
# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg.predict(X), color = 'blue')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
# in above poly_reg.fit_transform(X) is required bcz if we use X_poly instead this then it is wrong bcz if X will change after prediction
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Number of days')
plt.ylabel('Number of Corona cases')
X1=dataset.iloc[:,0:1].values;
y1=dataset.iloc[:,4].values;
print(y1)
poly_reg1 = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X1)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly, y1)
plt.scatter(X1, y1, color = 'red')
# plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.plot(X1, lin_reg_3.predict(poly_reg.fit_transform(X1)), color = 'blue')
# in above poly_reg.fit_transform(X) is required bcz if we use X_poly instead this then it is wrong bcz if X will change after prediction
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Number of days')
plt.ylabel('Number of Recovered')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, lin_reg_2.predict(X_grid), color = 'blue')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Number of days')
plt.ylabel('Number of Corona cases')
X_grid = np.arange(min(X1), max(X1), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X1, y1, color = 'red')
# plt.plot(X_grid, lin_reg_2.predict(X_grid), color = 'blue')
plt.plot(X_grid, lin_reg_3.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Number of days')
plt.ylabel('Number of Recovered')

plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[178]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[181]])))
print(lin_reg_3.predict(poly_reg.fit_transform([[179]])))
# # [827684.77504983]
# # [522067.4381997]
# # [852958.11001061]
# # [540728.61799884]
# # prediction of 23th july
# # [1237019.52354147]
# # [780339.44037567]
# #prediction of 24 th july
# [1271148.67578891]
# [831203.93585509]
# prediction of 26 th july using 180days
#  
# Decision Tree Regression

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # Importing the dataset
# dataset = pd.read_csv('nation_level_daily1.csv',index_col='index')
# X = dataset.iloc[:, 0:1].values
# y = dataset.iloc[:, 2].values
# print(X)
# print(y)
# # Training the Decision Tree Regression model on the whole dataset
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state = 0)
# regressor.fit(X, y)

# # Predicting a new result
# print(regressor.predict([[176]]))

# # Visualising the Decision Tree Regression results (higher resolution)
# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X, regressor.predict(X),color='blue')
# # plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (Decision Tree Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()