from flask import Flask, render_template,request
import tensorflow as tf 
from tensorflow import keras
import sys, os
from imageio import imread
import base64
import io
from scipy import misc
import numpy as np
global graph, model
from model import *
import datetime
import shutil




app = Flask(__name__)

# model = keras.models.load_model("digit_model_new.h5")

model, graph = load_model()

# model.load_weights('model_weights.h5')

model = keras.models.load_model('digit_model_new.h5')

@app.route("/")
def index():
	return render_template("index.html")



@app.route("/DigitRecognition", methods=["GET","POST"])
def digitRecognition():
	prediction = ''
	button_value ='none'

	# if request.method == "POST":
	# 	image_text  = request.form["test_image"]
	# 	image_text = image_text.split(",")[1]
	# 	image_data = imread(io.BytesIO(base64.b64decode(image_text)))
	# 	# image_data = image_data/255.0
	# 	misc.imsave('output.png',image_data)
	# 	image_data = misc.imread('output.png',mode='P')
	# 	image_data = np.invert(image_data)
	# 	image_data = misc.imresize(image_data,size=(28,28))
	# 	image_data = np.reshape(image_data,(1,28,28,1))
	# 	button_value = 'inline-block'

	# 	# prediction = model.predict(image_data)
	# 	# return render_template("DigitRecognition.html",predictions = prediction )

	# 	with graph.as_default():
	# 		prediction = np.argmax(model.predict(image_data))
			
	# 		return render_template("DigitRecognition.html",predictions = prediction, button_value = button_value)

		# with tf.Graph().as_default():
		# 	out = model.predict(image_data)
		# 	print(out)
		# 	print(np.argmax(out, axis=1)) 
		# 	prediction = np.array_str(np.argmax(out, axis=1))
		# image_data = image_data.reshape((28,28))
		# plt.imshow(image_data)
		# plt.show()
		# print(model.predict(image_data))
		# with open('test_image_save.png','wb') as fh:
		# 	fh.write(base64.decodebytes(image_text.encode()))

	return render_template("DigitRecognition.html",preidctions = prediction,button_value = button_value )


@app.route("/predict",methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		image_text  = request.form["image"]
		image_text = image_text.split(",")[1]
		image_data = imread(io.BytesIO(base64.b64decode(image_text)))
		# image_data = image_data/255.0
		misc.imsave('output.png',image_data)
		image_data = misc.imread('output.png',mode='P')
		image_data = np.invert(image_data)
		image_data = misc.imresize(image_data,size=(28,28))
		image_data = np.reshape(image_data,(1,28,28,1))
		with graph.as_default():
			prediction = np.argmax(model.predict(image_data))
			print("Good")

			return str(prediction)


@app.route('/function', methods = ['GET','POST'])
def fun():
	return render_template('function.html')

@app.route('/move' , methods = ['GET'])
def move_test_files():
	right_ans = request.args.get('right_ans')
	name = "./test_images/output" + str(datetime.datetime.now())+"target("+right_ans +").png"
	shutil.copyfile("output.png", name)
	return "thank you"


if __name__ == "__main__":
	app.run(debug=True)
	
