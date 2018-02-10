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

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

x_in = np.random.randn(1,4)
x_in[0][0] = 1
x_in[0][1] = 3
x_in[0][2] = 1	
x_in[0][3] = 4

pred = loaded_model.predict(x_in)

print(np.argmax(pred))