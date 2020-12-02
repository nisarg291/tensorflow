import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import matplotlib.dates as mdates
# import joblib

# ### so that u dont have warnings
# from warnings import filterwarnings
# filterwarnings('ignore')

##Step1: Load Dataset
def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter


dataframe = pd.read_csv("individual_stocks_5yr/2020pricing/TCS.csv")
print(dataframe.head())
dataframe['Date']=pd.to_datetime(dataframe['Date'])
#Step2: Split into training and test data
# dataframe["Date"] = pd.to_datetime(dataframe["Date"]).dt.strftime("%Y%m%d")
print( dataframe.head() )
X=dataframe.iloc[:,0].values
y = dataframe.iloc[:, 8].values
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

plt.plot_date(X,y,'-')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Encoding categorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# # labelencoder_X=LabelEncoder()
# # X[:,3]=labelencoder_X.fit_transform(X[:,3])
# # onehotencoder=OneHotEncoder(categories=[3])
# # X=onehotencoder.fit_transform(X).toarray()
# # print(X)#


# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# # Training the Multiple Linear Regression model on the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train.reshape(-1,1), y_train)

# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
# print("last prediction ",y_pred[len(y_pred)-12:len(y_pred)-2])
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# plt.plot(range(0,len(y_test)),y_test,'o')
# plt.plot(range(0,len(y_test)),y_pred,'x')
# plt.grid()
# plt.show()

# #Building the optimal model using backword elimination

# import statsmodels.formula.api as sm
# X=np.append(arr=np.ones((50,3)).astype(int),values=X,axis=1)
# X_opt=X[:,[3,4,5,6,7]]
# reg_OLS=sm.OLS(y,X_opt)
# reg_OLS.summary()
# X_opt=X[:,[0,3]]
# reg_OLS=sm.OLS(y,X_opt)
# reg_OLS.summary()








# # Predicting a new result
# # print(regressor.predict([[6.5]]))

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = dataframe.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(dataframe)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:3750,:]
valid = dataset[3750:4000,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=2)
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
print(closing_price)

train = new_data[:3750]
valid = new_data[3750:4000]

plt.plot(train['Close'])
# plt.plot(closing_price)
valid['Predictions'] = closing_price
print(closing_price)
plt.plot(valid['Predictions'])
plt.show()

future = model.make_future_dataframe(periods=365) #we need to specify the number of days in future
prediction = model.predict(future)
model.plot(prediction)
plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()

model.plot_components(prediction)
plt.show()