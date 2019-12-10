import os
import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(os.path.dirname(os.path.dirname(dir_path)), 'Fantasy-Premier-League', 'data')

historical_seasons = ['2016-17', '2017-18', '2018-19']
current_season = ['2019-20']

for season in current_season:
	with open (os.path.join(data_path, season, 'players_raw.csv')) as players_raw:
		players_raw_df = pd.read_csv(players_raw)
		for player in os.listdir(os.path.join(data_path, season, 'players')):
			with open(os.path.join(data_path, season, 'players', player, 'gw.csv')) as gws:
				df = pd.read_csv(gws)

				# Add features from other sources
				# player position i.e. element_type
				position = players_raw_df.loc[players_raw_df['id'] == df['element'][0]]['element_type']
				if position.shape[0] == 0:
					print(df['id'][0])
#					print(players_raw_df.loc[players_raw_df['first_name'] == 'Islam']['id'])
				position = position.iat[0]
#				print(position)
				df.insert(0, 'position', position)


				# team xG and xGA
				# player xG and xA
				# opponent cumulative xG and xGA


				os.mkdir(os.path.join(os.path.dirname(dir_path), 'data', season, 'players', player))
				df.to_csv(os.path.join(os.path.dirname(dir_path), 'data', season, 'players', player, 'gw.csv'))

