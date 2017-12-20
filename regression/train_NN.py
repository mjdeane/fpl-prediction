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
    x = np.array(map(lambda z: map(float, z), x))
    print(x.shape)
with open('../data/attackers_y.csv', 'r') as csv_y:
    r = csv.reader(csv_y, delimiter=' ')
    y = np.array(list(r))
    y.resize(y.shape[0])
    y = np.array(map(float,y))

def baseline_model():
    model = Sequential()
    model.add(Dense(85, input_dim=85, activation='relu', kernel_initializer='normal'))
    model.add(Dense(20, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

#    model.fit(x,y, batch_size = 20, epochs = 2, validation_split = 0.2)
    return model


#baseline_model()







seed = 5
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)

#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(pipeline, x, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
