

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  save_model
import pandas as pd
import numpy as np
from scipy import misc


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training Data : {} " .format(x_train.shape))
print("Training Target : {}".format(y_train.shape))

"""**It is 28 x 28 images of 60000 handwritten images. Let's take a look at the data**"""

# plt.imshow(x_train[0])
# plt.show()

# Stadardize the data
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train/255.0
x_test = x_test/255.0

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation = "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation = "softmax"))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train,y_train,epochs=2)

print(model.evaluate(x=x_test,y=y_test))

model.save_model('digitModel.h5')



