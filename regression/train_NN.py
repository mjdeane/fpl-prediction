import sys
sys.path.append('..')
import os
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scraping import api_utils

dir_path = os.path.dirname(os.path.realpath(__file__))

x = [];
y = [];
os.path.join(os.path.dirname(dir_path),'data','player_input_vectors_')
# here we take attackers_x and attackers_y from csv to np.array
# need to get new data instead here in (SAME SHAPE/DIFFERENT SHAPE ??)
with open(os.path.join(os.path.dirname(dir_path),'data','attackers_x.csv'), 'r') as csv_x:
    r = csv.reader(csv_x, delimiter=' ')
    x = np.array(list(r))
    x = x.astype(float)
with open(os.path.join(os.path.dirname(dir_path),'data','attackers_y.csv'), 'r') as csv_y:
    r = csv.reader(csv_y, delimiter=' ')
    y = np.array(list(r))
    y.resize(y.shape[0])
    y = y.astype(float)

x0 =[]
y0=[]
#take test data from csvs
# get new test data in (SAME SHAPE/DIFFERENT SHAPE)
with open(os.path.join(os.path.dirname(dir_path),'data','attackers2_x.csv'), 'r') as csv_x:
    r = csv.reader(csv_x, delimiter=' ')
    x0 = np.array(list(r))
    x0 = x0.astype(float)
with open(os.path.join(os.path.dirname(dir_path),'data','attackers2_y.csv'), 'r') as csv_y:
    r = csv.reader(csv_y, delimiter=' ')
    y0 = np.array(list(r))
    print(y0.shape)
    print(y0.shape[0])
    y0.resize(y0.shape[0])
    print(y0.shape)
    print(str(y0))
    y0 = y0.astype(float)

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
x0 = scaler.transform(x0)
model.fit(x,y, batch_size = 50, epochs = 5, validation_data=(x0,y0))#, validation_split = 0.25)

names = api_utils.getIdToNameDict()
#print(names)
(Z,ids) = api_utils.getAllPlayerInputVectors()
Z = scaler.transform(Z)
prediction = model.predict(Z)

#
with open(os.path.join(os.path.dirname(dir_path),'data','gameweek_'+ str(api_utils.getNextGameweek()) + '_predictions.csv'),'w') as outfile:
    w = csv.writer(outfile, delimiter=' ')
    j = 0
    for i in ids:
        if i in names:
            row = names[i], prediction[j][0]
            j+=1
            w.writerow(row)


print('EVALUATION')

print(model.evaluate(x=x0,y=y0,batch_size=20))

#test_vector = api_utils.getPlayerInputVector(279)
#print(test_vector)
#test_vector = scaler.transform(test_vector)
#print(test_vector)
#prediction = model.predict(test_vector)
#print(prediction)

