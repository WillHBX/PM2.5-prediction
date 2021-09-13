from math import sqrt
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.backend import expand_dims
from tensorflow.config import list_physical_devices
from tensorflow.config.experimental import set_memory_growth
import json

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], enable=True)

data = np.load('/content/drive/MyDrive/WanHua_PM2d5_data.npz')

train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']



train_x = expand_dims(train_x,axis = -1)
train_y = expand_dims(train_y,axis = -1)
test_x = expand_dims(test_x,axis = -1)
test_y = expand_dims(test_y,axis = -1)


print(np.shape(train_x))
print(np.shape(train_y))
print(np.shape(test_x))
print(np.shape(test_y))


# design network
model = Sequential()
# model.add(LSTM(30, input_shape = (36,1),return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_x, train_y, epochs=100, batch_size=200, validation_split=0.3, verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_x)

pyplot.figure(figsize=(20,10))
pyplot.plot(test_y, 'o', label = 'test_y')
pyplot.plot(yhat, 'o', label = 'predict_y')
pyplot.legend()
pyplot.show()

pyplot.figure(figsize=(20,10))
pyplot.plot(test_y, label = 'test_y')
pyplot.plot(yhat, label = 'predict_y')
pyplot.legend()
pyplot.show()
# calculate RMSE

mae = mean_absolute_error(test_y, yhat)
rmse = sqrt(mean_squared_error(test_y, yhat))


print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)




