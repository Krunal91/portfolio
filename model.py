from tensorflow import keras
import tensorflow as tf 


def load_model():

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


	graph = tf.get_default_graph()

	return model, graph