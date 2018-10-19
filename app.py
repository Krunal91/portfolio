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
cat_model = tf.keras.models.load_model("./models/scratch_model.h5")

# model.load_weights('model_weights.h5')

model = keras.models.load_model('./models/digit_model_new.h5')

@app.route("/")
def index():
	return render_template("index.html")



@app.route("/DigitRecognition", methods=["GET","POST"])
def digitRecognition():
	return render_template("DigitRecognition.html")


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
			return str(prediction)




@app.route('/move' , methods = ['GET'])
def move_test_files():
	right_ans = request.args.get('right_ans')
	name = "./test_images/output" + str(datetime.datetime.now())+"target("+right_ans +").png"
	shutil.copyfile("output.png", name)
	return "thank you"


@app.route("/catvsdog",methods=["POST","GET"])
def catvsdog():

	if request.method == "POST":
		print("this is working")
		f = request.files.get('pic')
		if f:
			data = imread(f.read())
			print(data.shape)
			f.save("./"+f.filename)
			# image_data = imread(io.BytesIO(f.read()))

		else:
			print("No file")

		return render_template('catvsdog.html')

	return render_template('catvsdog.html')

@app.route("/catpredict",methods=["GET","POST"])
def catpredict():
	if request.method == 'POST':
		image_text  = request.form["image"]
		image_text = image_text.split(",")[1]
		image_data = imread(io.BytesIO(base64.b64decode(image_text)))
		# image_data = image_data/255.0
		# misc.imsave('cat.png',image_data)
		# image_data = misc.imread('output.png',mode='RGB')
		# image_data = np.invert(image_data)
		image_data = misc.imresize(image_data,size=(150,150))
		image_data = np.expand_dims(image_data,axis=0)
		print(image_data.shape)
		with graph.as_default():
			predictions = cat_model.predict(image_data)
			if predictions[[0]] == 0:
				return "It is a Cat!"
			else:
				return "It is a Dog!"

	return "Something went Wrong"


@app.route("/temp")
def temp():
	return render_template("temp.html")


if __name__ == "__main__":
	app.run(debug=True)
	
