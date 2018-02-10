import numpy as np
import flask

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def prepare_image(image, target):
	
	# convert actual values to classes

	return image

@app.route("/")
def entry():
	return "Welcome to te zero research travel api"

@app.route("/predict", methods=["GET"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	# ensure an image was properly uploaded to our endpoint
	data = flask.request.args.to_dict()
		
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()