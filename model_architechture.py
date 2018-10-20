from tensorflow import keras
from scipy import misc
import numpy as np

def digit_model_summary(s):
	with open("digit_model_summary.txt","a") as f:
		print(s,file =f)

def cat_model_summary(s):
	with open("cat_model_summary.txt","a") as f:
		print(s,file =f)	


digit_model = keras.models.load_model("./models/digit_model_new.h5")
cat_model = keras.models.load_model("./models/scratch_model.h5")
digit_model.summary(print_fn = digit_model_summary)
cat_model.summary(print_fn = cat_model_summary)


