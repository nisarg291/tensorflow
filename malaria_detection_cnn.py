import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()



X_train  = train.drop(['label'],axis=1).values
y_train  = train['label'].values

X_test  = test.drop(['label'],axis=1).values
y_test  = test['label'].values


X_test.shape, y_test.shape

import matplotlib.pyplot as plt

index = 100

plt.imshow(X_train[index].reshape(50,50),cmap='gray')
print(y_train[index])

X_train = X_train.reshape(train.shape[0],50,50,1).astype('float32')
X_train = X_train / 255.0

X_test = X_test.reshape(test.shape[0],50,50,1).astype('float32')
X_test = X_test / 255.0

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("X_train shape", X_train.shape)
print("Y_train shape", y_train.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


model = Sequential()

model.add(Conv2D(filters=16,kernel_size=3,padding="same",activation="relu",input_shape=(50,50,1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(200,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=50,epochs=20,verbose=1)

predictions = model.evaluate(X_test,y_test)

index = 4001
import numpy as np
plt.imshow(X_test[index].reshape(50,50),cmap='gray')
print("Actual",y_test[index])

print("Predicted", model.predict([[X_test[index]]]))

