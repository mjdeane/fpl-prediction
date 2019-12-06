import os
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed

dir_path = os.path.dirname(os.path.realpath(__file__))

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
			         'attempted_passes',
			         'big_chances_created',
			         'big_chances_missed',
			         'bonus',
			         'bps',
			         'clean_sheets',
			         'clearances_blocks_interceptions',
			         'completed_passes',
			         'creativity',
			         'dribbles',
			         'ea_index',
			         #'element',
			         'errors_leading_to_goal',
			         'errors_leading_to_goal_attempt',
			         #'fixture',
			         'fouls',
			         'goals_conceded',
			         'goals_scored',
			         'ict_index',
			         #'id',
			         'influence',
			         'key_passes',
			         #'kickoff_time',
			         #'kickoff_time_formatted',
			         #'loaned_in',
			         #'loaned_out',
			         'minutes',
			         'offside',
			         'open_play_crosses',
			         'opponent_team',
			         'own_goals',
			         'penalties_conceded',
			         'penalties_missed',
			         'penalties_saved',
			         'recoveries',
			         'red_cards',
			         'round',
			         'saves',
			         'selected',
			         'tackled',
			         'tackles',
			         'target_missed',
			         'team_a_score',
			         'team_h_score',
			         'threat',
			         'transfers_balance',
			         'transfers_in',
			         'transfers_out',
			         'value',
			         #'was_home',
			         'winning_goals',
			         'yellow_cards']]
			if df.shape[0] < 2:
				continue
			#df.was_home.map({True: 1, False: 0})
			x.append(df.values.reshape(1,-1,46)[:,:-1,:])
			y.append(df.total_points.values.reshape(1, -1, 1)[:,1:,:])
	return x, y

x_train, y_train = get_player_data(os.path.join(os.path.dirname(dir_path), 'data', '2016-17', 'players'), mids_set_16)
x_val, y_val = get_player_data(os.path.join(os.path.dirname(dir_path), 'data', '2017-18', 'players'), mids_set_17)

def train_generator():
	while True:
		for i in range(0,len(x_train)):
			yield x_train[i], y_train[i]

def val_generator():
	while True:
		for i in range(0,len(x_val)):
			yield x_val[i], y_val[i]

# try a simple lstm with the data
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(None,46)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit_generator(train_generator(),
                    validation_data=val_generator(),
                    steps_per_epoch=len(x_train),
                    validation_steps=len(x_val),
                    epochs=1)

model.save('one_gw_prediction_lstm.h5')