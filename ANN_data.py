# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:06:57 2019

@author: mluth
"""
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


#importing the dataset
dataset = pd.read_csv(r"C:\Users\mluth\Desktop\A- Z Machine Learing\32 Convolutional Neural Networks\Artificial_Neural_Networks\Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#importing the labelling the labelencoder and Onehotencoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
le_2 = LabelEncoder()
X[:,1] = le_1.fit_transform(X[:,1])
X[:,2] = le_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features="all")
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#initialize the cnn
classifier = Sequential()

#inner dense layer

classifier.add(Dense(output_dim = 10, activation = 'relu', input_dim = 11))

classifier.add(Dense(output_dim = 1, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the cnn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train,y_train,epochs=100)
#classifier.fit_generator(training_set,steps_per_epoch = 80,epochs = 10,validation_data = test_set,validation_steps=80)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix as cf
c_matrix = cf(y_test,y_pred)