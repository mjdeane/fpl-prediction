import os
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(os.path.dirname(os.path.dirname(dir_path)), 'Fantasy-Premier-League', 'data')

mids_set_16 = set([])
mids_set_17 = set([])

# find only midfielders
#with open(os.path.join(os.path.dirname(dir_path), 'data', '2016-17', 'players_raw.csv')) as players:
#	df = pd.read_csv(players)
#	for i in df[np.logical_or(df['element_type'] == 2 , df['element_type'] ==2)].id:
#		mids_set_16.add(i)
#with open(os.path.join(os.path.dirname(dir_path), 'data', '2017-18', 'players_raw.csv')) as players:
#	df = pd.read_csv(players)
#	for i in df[np.logical_or(df['element_type'] == 2 , df['element_type'] ==2)].id:
#		mids_set_17.add(i)

#print(len(mids_set_16))
#print(mids_set_16)
#print(len(mids_set_17))
#print(mids_set_17)


def get_player_data(players_dir, mids_set):
	x = []
	y = []

	for player in os.listdir(players_dir):
		with open(os.path.join(players_dir, player, 'gw.csv')) as gws:
			df = pd.read_csv(gws)
#			if df.element[0] not in mids_set:
#				continue
			df = df[['total_points',
			         'assists',
			         #'attempted_passes',
			         #'big_chances_created',
			         #'big_chances_missed',
			         'bonus',
			         'bps',
			         'clean_sheets',
			         #'clearances_blocks_interceptions',
			         #'completed_passes',
			         'creativity',
			         #'dribbles',
			         #'ea_index',
			         #'element',
			         #'errors_leading_to_goal',
			         #'errors_leading_to_goal_attempt',
			         #'fixture',
			         #'fouls',
			         'goals_conceded',
			         'goals_scored',
			         'ict_index',
			         #'id',
			         'influence',
			         #'key_passes',
			         #'kickoff_time',
			         #'kickoff_time_formatted',
			         #'loaned_in',
			         #'loaned_out',
			         'minutes',
			         #'offside',
			         #'open_play_crosses',
			         'opponent_team',
			         'own_goals',
			         #'penalties_conceded',
			         'penalties_missed',
			         'penalties_saved',
			         #'recoveries',
			         'red_cards',
			         'round',
			         'saves',
			         'selected',
			         #'tackled',
			         #'tackles',
			         #'target_missed',
			         'team_a_score',
			         'team_h_score',
			         'threat',
			         'transfers_balance',
			         'transfers_in',
			         'transfers_out',
			         'value',
			         'was_home',
			         #'winning_goals',
			         'yellow_cards']]
			if df.shape[0] < 2:
				continue
			if df.shape[0] < 38:
				zeros_df = pd.DataFrame(np.zeros((38 - df.shape[0],28)), columns=df.columns)
				df = pd.concat([zeros_df, df])
			df.was_home.map({'True': 1, 'False': 0})
			x.append(df.values.astype(float).reshape(1,38,28)[:,:-1,:])
			y.append(df.total_points.values.astype(float).reshape(1, 38, 1)[:,1:,:])
	x = np.stack(x).reshape(-1,37,28)
	y = np.stack(y).reshape(-1,37,1)
	return x, y

x_train, y_train = get_player_data(os.path.join(data_path '2016-17', 'players'), mids_set_16)
x_train0, y_train0 = get_player_data(os.path.join(data_path, '2017-18', 'players'), mids_set_16)
x_val, y_val = get_player_data(os.path.join(data_path, '2018-19', 'players'), mids_set_17)

x_train = np.concatenate((x_train, x_train0))
y_train = np.concatenate((y_train, y_train0))

#def train_generator():
#	while True:
#		for i in range(0,len(x_train)):
#			yield x_train[i], y_train[i]

#def val_generator():
#	while True:
#		for i in range(0,len(x_val)):
#			yield x_val[i], y_val[i]

# try a simple lstm with the data
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(None,28)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(loss='mean_squared_error', optimizer='adam')

#model.fit_generator(train_generator(),
#                    validation_data=val_generator(),
#                    steps_per_epoch=len(x_train),
#                    validation_steps=len(x_val),
#                    epochs=1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1,28)).reshape(x_train.shape)
x_val = scaler.transform(x_val.reshape(-1,28)).reshape(x_val.shape)

model.fit(x=x_train,
	      y=y_train,
          validation_data=(x_val,y_val),
          batch_size=32,
          epochs=10)

joblib.dump(scaler, 'standard_scaler.joblib')
model.save('one_gw_prediction_lstm.h5')