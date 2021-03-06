import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from utils import api_utils

dir_path = os.path.dirname(os.path.realpath(__file__))

model = load_model(os.path.join(dir_path, 'one_gw_prediction_lstm.h5'))
scaler = joblib.load(os.path.join(dir_path, 'standard_scaler.joblib'))

GW = api_utils.getNextGameweek()

x = []
players_dir = os.path.join(os.path.dirname(dir_path), 'data', '2019-20', 'players')
players = os.listdir(players_dir)

num_features = 33

for player in players:
	with open(os.path.join(players_dir, player, 'gw.csv')) as gws:
		df = pd.read_csv(gws)
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
			         'defender',
			         #'dribbles',
			         #'ea_index',
			         #'element',
			         #'errors_leading_to_goal',
			         #'errors_leading_to_goal_attempt',
			         #'fixture',
			         'forward',
			         #'fouls',
			         'goalkeeper',
			         'goals_conceded',
			         'goals_scored',
			         #'ict_index',
			         #'id',
			         'influence',
			         #'key_passes',
			         #'kickoff_time',
			         #'kickoff_time_formatted',
			         #'loaned_in',
			         #'loaned_out',
			         'midfielder',
			         'minutes',
			         #'offside',
			         #'open_play_crosses',
			         'opponent_cum_xG',
			         'opponent_cum_xGA',
			         'opponent_cum_xpts',
			         'opponent_team',
			         'own_goals',
			         #'penalties_conceded',
			         'penalties_missed',
			         'penalties_saved',
			         #'position',
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
			         #'transfers_balance',
			         'transfers_in',
			         'transfers_out',
			         'value',
			         'was_home',
			         #'winning_goals',
			         'yellow_cards']]
		if df.shape[0] < GW - 1:
			zeros_df = pd.DataFrame(np.zeros((GW - 1 - df.shape[0],num_features)), columns=df.columns)
			df = pd.concat([zeros_df, df])
		if df.shape[0] > GW -1:
			df = df[:GW]
		df.was_home.map({'True': 1, 'False': 0})
		array = np.nan_to_num(df.values.astype(float))
		x.append(scaler.transform(array).reshape(1,-1,num_features))

x = np.stack(x).reshape(-1, GW-1, num_features)

predictions = [i[-1][0] for i in model.predict(x)]

output = pd.DataFrame(zip(players, predictions), columns = ['player', 'predicted score'])
output = output.sort_values(by='predicted score',ascending=False).round(decimals=2)

output.to_csv(os.path.join(os.path.dirname(dir_path), 'data', 'predictions', 'predictions_{}.csv'.format(GW)))