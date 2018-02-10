import numpy as np
import pandas as pd
import sklearn
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Activation, Average, Dropout
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

tsize = 18047

x_train = np.random.randn(tsize, 4)
y_train = np.random.randint(85, size=(tsize, 1))

df = pd.read_csv("TravelDatabase2.csv")
#df['spend'] = df['spend'].astype('float64').isnull(0.0)

for i in range(tsize):
	x_train[i][0] =  df['quarter'][i]
	x_train[i][1] = df['Age'][i]
	if df['Sex'][i] == '#NULL!':
		x_train[i][2] = 0.5
	else: 
		x_train[i][2] = df['Sex'][i]
	x_train[i][3] = df['duration'][i]
	#x_train[i][4] = float(df['spend'][i])
	y_train[i] = df['country'][i] - 10
y_train = to_categorical(y_train, num_classes = 85)


'''x_test = np.random.randn(100, 20)
y_test = to_categorical(np.random.randint(80, size=(100, 1)), num_classes=80)'''

model1 = Sequential()
model1.add(Dense(64, activation='relu', input_dim=4))
#model.add(Dropout(0.2))
model1.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))
model1.add(Dense(85, activation='softmax'))

model1.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model1.fit(x_train, y_train,
          epochs=30,
          batch_size=128)

model_json = model1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model1.save_weights("model.h5")




