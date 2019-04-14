# -*- coding: utf-8 -*-
"""
Spyder Editor

manish luthyagi
"""

# importing the libraries 

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initialize the cnn
classifier = Sequential()

 # convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64 ,64, 3),activation = 'relu')

 # pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

 # Flattening
classifier.add(Flatten())

 # Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the cnn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#part2- fitting the cnn to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',target_size=(64,64),batch_size=32,class_mode='binary')


#training our network

from Ipython.display import display
form PIL import Image

classifier.fit_generator(training_set,steps_per_epoch = 8000,epochs = 10,validation_data=test_set,validation_steps=800)


# Testing 
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('random.jpg',target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
training_set.class_indices


if result[0][0] >= 0.5
	prediction = 'dog'
else:
	prediction = 'cat'
print(prediction)
