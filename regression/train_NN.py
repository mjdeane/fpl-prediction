import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x = [];
y = [];

with open('../data/attackers_x.csv', 'r') as csv_x:
    r = csv.reader(csv_x, delimiter=' ')
    x = np.array(list(r))
    x = x.astype(float)
with open('../data/attackers_y.csv', 'r') as csv_y:
    r = csv.reader(csv_y, delimiter=' ')
    y = np.array(list(r))
    y.resize(y.shape[0])
    y = y.astype(float)

def baseline_model():
    model = Sequential()
    model.add(Dense(85, input_dim=85, activation='relu', kernel_initializer='normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

model = baseline_model()
scaler = StandardScaler()
x = scaler.fit_transform(x)
model.fit(x,y, batch_size = 50, epochs = 5, validation_split = 0.25)


