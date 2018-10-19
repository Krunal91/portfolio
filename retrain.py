from tensorflow import keras
import os
from scipy import misc


model = keras.models.load_model('digit_model_new.h5')

files = os.listdir("./test_images")
data = []

for file in files:
	print(file)
	image_data =  misc.imread(file)
	print(image_data.shape)
	break


