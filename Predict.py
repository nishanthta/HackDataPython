import numpy as np
from keras.models import model_from_json

num_queries = 5

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model1.load_weights("model1.h5")

json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model2.load_weights("model2.h5")

json_file = open('model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model3 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model3.load_weights("model3.h5")

x_in = np.random.randn(1,5)
x_in[0][0] = 1 # Quarter
x_in[0][1] = 2 # Age
x_in[0][2] = 1	# Sex
x_in[0][3] = 2 # Duration
x_in[0][4] = 1 # Budget

pred1 = loaded_model1.predict(x_in)
pred2 = loaded_model2.predict(x_in)
pred3 = loaded_model3.predict(x_in)

pred = (pred1 + pred2 + pred3)/3

i = iter(pred)
pred_dict = {pred[0][i]: i for i in range(0, 85)}

pred_dict = sorted(pred_dict.items())

prefs = []

for _,value in pred_dict:
	prefs.append(value)

prefs.reverse()	
data = {}

with open('CountryDB.txt') as f:
	cnt = 1
	line = f.readline()
	while line:
		data[cnt] = line.strip()
		line = f.readline()
		cnt += 1

for i in range(num_queries):
	print(data[prefs[i]])