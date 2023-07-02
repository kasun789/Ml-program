import tensorflow as tf
from tensorflow import keras 
import pandas as pd
import numpy as np
from keras import datasets,layers,models
from keras.utils import to_categorical

dataFolder = r'C:\Users\kasun\Downloads\adaptediris.csv'
df = pd.read_csv(dataFolder)
df.head()
print(df.drop_duplicates(subset=['variety']))

X=df.iloc[:,0:4].values
print(X)
y= df.iloc[:,4].values
print(y)

from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
print(y1)

Y = pd.get_dummies(y1).values
print(Y)

print(Y[51:56])
print(Y[len(Y)-6:len(Y)-1])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

print(x_train[0:5])
print(y_train[0:5])
print(x_test[0:5])
print(y_test[0:5])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])

from PIL import Image

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=50,epochs=100)

loss,accuracy = model.evaluate(x_test,y_test,verbose=0)

print("Test loss",loss)
print("Test accuracy",accuracy)

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred[5])

random_set_of_feature_values = [[4.0,2.4,5.7,3.4]]
print(model.predict(random_set_of_feature_values))


