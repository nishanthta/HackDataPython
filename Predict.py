import numpy as np
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
x_in[0][0] = 1
x_in[0][1] = 3
x_in[0][2] = 1	
x_in[0][3] = 4
x_in[0][4] = 5

pred1 = loaded_model1.predict(x_in)
pred2 = loaded_model2.predict(x_in)
pred3 = loaded_model3.predict(x_in)

pred = (pred1 + pred2 + pred3)/3

print(pred)

i = iter(pred)
pred_dict = {pred[0][i]: i for i in range(0, 85)}

pred_dict = sorted(pred_dict.items())

prefs = []

for _,value in pred_dict:
	prefs.append(value)

print(prefs)
