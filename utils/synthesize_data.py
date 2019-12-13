import os
import numpy as np
import pandas as pd
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(os.path.dirname(os.path.dirname(dir_path)), 'Fantasy-Premier-League', 'data')

historical_seasons = ['2016-17', '2017-18', '2018-19']
current_season = ['2019-20']

for season in current_season:
	team_metrics = {}
	with open(os.path.join(os.path.dirname(dir_path), 'data', season, 'team_to_id.json')) as jsonfile:
		team_to_id = json.load(jsonfile)
		for team in os.listdir(os.path.join(data_path, season, 'understat')):
			if team == 'understat_player.csv' or team == 'understat_team.csv':
				continue
			team_name = team.split('_', maxsplit=1)[1].split('.')[0]
			df = pd.read_csv(os.path.join(data_path, season, 'understat', team))
			print(season)
			print(team)
			print(team_name)
			print(df)
			team_metrics[team_to_id[team_name]] = df[['xG', 'xGA', 'xpts']].cumsum()
	with open (os.path.join(data_path, season, 'players_raw.csv')) as players_raw:
		players_raw_df = pd.read_csv(players_raw)
		for player in os.listdir(os.path.join(data_path, season, 'players')):
			with open(os.path.join(data_path, season, 'players', player, 'gw.csv')) as gws:
				df = pd.read_csv(gws)

				# Add features from other sources
				# player position i.e. element_type
				position = players_raw_df.loc[players_raw_df['id'] == df['element'][0]]['element_type']
				#TODO: handle gracefully when id is not found
				position = position.iat[0]
				goalkeeper = 1 if position == 1 else 0
				defender = 1 if position == 2 else 0
				midfielder = 1 if position == 3 else 0
				forward = 1 if position == 4 else 0

#				print(position)
				df.insert(0, 'goalkeeper', goalkeeper)
				df.insert(0, 'defender', defender)
				df.insert(0, 'midfielder', midfielder)
				df.insert(0, 'forward', forward)


				# team xG and xGA

#				df['opponent_cum_xG'] = df.apply(lambda row: team_metrics[str(row.id)].iat[row.round-1].xG, axis=1)
#				print(df.opponent_cum_xG)
				opponent_cum_xG = [0]
				opponent_cum_xGA = [0]
				opponent_cum_xpts = [0]
				for gameweek in df.iloc[1:].iterrows():
					gameweek=gameweek[1]
					opponent = team_metrics[str(gameweek.loc['opponent_team'])]
					opponent_cum_xG.append(opponent['xG'].iat[gameweek.loc['round']-2])
					opponent_cum_xGA.append(opponent['xGA'].iat[gameweek.loc['round']-2])
					opponent_cum_xpts.append(opponent['xpts'].iat[gameweek.loc['round']-2])

				df.insert(0, 'opponent_cum_xpts', opponent_cum_xpts)
				df.insert(0, 'opponent_cum_xGA', opponent_cum_xGA)
				df.insert(0, 'opponent_cum_xG', opponent_cum_xG)

				# player xG and xA
				# opponent cumulative xG and xGA

				#TODO: check if player directory exists and if not create it - this will be breaking
				#os.mkdir(os.path.join(os.path.dirname(dir_path), 'data', season, 'players', player))
				df.to_csv(os.path.join(os.path.dirname(dir_path), 'data', season, 'players', player, 'gw.csv'))