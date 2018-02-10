import numpy as np
import flask
import pandas as pd
import sklearn
from keras.models import Model, Input, Sequential, model_from_json
from keras.layers import Dense, Activation, Average, Dropout
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def load_model():

	global loaded_model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")

def prepare_image(image, target):
	
	# convert actual values to classes

	return image
	
@app.route("/")
def entry():
	return "Welcome to te zero research travel api"

@app.route("/predict", methods=["GET"])
def predict():
	data = flask.request.args.to_dict()
		
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	#load_model()
	app.run()