import requests
import json
import csv
import argparse
import numpy as np
import os

FPL_URL = "https://fantasy.premierleague.com/drf/"
PLAYER_SUMMARY_SUBURL = "element-summary/"
PLAYER_SUMMARY_URL = FPL_URL + PLAYER_SUMMARY_SUBURL
PLAYERS_INFO_SUBURL = "bootstrap-static"
PLAYERS_INFO_URL = FPL_URL + PLAYERS_INFO_SUBURL



def getNextGameweek():
    r = requests.get(PLAYERS_INFO_URL)
    info = r.json()
    return int(info['next-event'])

def getAllPlayersInfo():
    r = requests.get(PLAYERS_INFO_URL)
    info = r.json()
#    print(info.keys())
#    print(info['elements'][246]['id'])
    return list(filter( lambda x: int(x['element_type'])>2,info['elements']))

def getPlayerInfo(player_id):
    r = requests.get(PLAYER_SUMMARY_URL + str(player_id))
    json_response = r.json()
    info = json_response
    return info

def getPlayerInputVector(player_id):
    info = getPlayerInfo(player_id)
    gw = getNextGameweek()
    vector = np.empty(85)
    for i in range(gw-6,gw-1):
#        print(info['history'][i]['total_points'])
        if (len(info['history']) < gw-1):
            return np.zeros(85)
        vector[(i-gw+6)*17] = info['history'][i]['total_points']
        vector[(i-gw+6)*17+1] = info['history'][i]['goals_scored']
        vector[(i-gw+6)*17+2] = info['history'][i]['assists']
        vector[(i-gw+6)*17+3] = info['history'][i]['minutes']
        vector[(i-gw+6)*17+4] = info['history'][i]['bps']
        vector[(i-gw+6)*17+5] = info['history'][i]['goals_conceded']
        vector[(i-gw+6)*17+6] = info['history'][i]['influence']
        vector[(i-gw+6)*17+7] = info['history'][i]['creativity']
        vector[(i-gw+6)*17+8] = info['history'][i]['threat']
        vector[(i-gw+6)*17+9] = info['history'][i]['ict_index']
        vector[(i-gw+6)*17+10] = info['history'][i]['selected']
        vector[(i-gw+6)*17+11] = info['history'][i]['transfers_balance']
        vector[(i-gw+6)*17+12] = info['history'][i]['red_cards']
        vector[(i-gw+6)*17+13] = info['history'][i]['yellow_cards']
        vector[(i-gw+6)*17+14] = info['history'][i]['saves']
        vector[(i-gw+6)*17+15] = info['history'][i]['bonus']
        vector[(i-gw+6)*17+16] = info['history'][i]['clean_sheets']
#    print(vector)
    vector = vector.astype(float)
#    print(vector)
    vector = vector.reshape((1,85))
    return vector

def writeAllPlayerInputVectors():
    info = getAllPlayersInfo()
    ids = np.empty(len(info))
    Z = np.empty((len(info),85))
    j = 0
    for p in info:
        Z[j]= getPlayerInputVector(int(p['id']))
        ids[j] = int(p['id'])
        j += 1
    with open('../data/player_input_vectors_'+str(getNextGameweek()), 'w') as outfile:
        w = csv.writer(outfile, delimiter=' ')
        w.writerow(ids)
        for z in Z:
            w.writerow(z)

def getAllPlayerInputVectors():
    if not os.path.isfile('../data/player_input_vectors_'+str(getNextGameweek())):
        writeAllPlayerInputVectors()
    with open('../data/player_input_vectors_'+str(getNextGameweek()),'r') as infile:
        r = csv.reader(infile, delimiter=' ')
        l =  list(r)
        ids = np.array(l[0]).astype(float)
        Z = np.array(l[1:]).astype(float)
        return(Z,ids)

def getIdToNameDict():
    info = getAllPlayersInfo()
    names = dict([])

    for p in info:
        names[p['id']] = p['web_name']

#    print(names.keys())

    return names


#info = getAllPlayersInfo()[393]
#print(info['element_type'])
#print(list(info.keys()))
#print(info['history'][10])
#print(info['explain'])


#        line = tree.xpath('//h3[@data-ng-bind="title"]/text() | //p[@data-ng-bind="position"]/text()\
#                           | //p[@data-ng-bind="team"]/text() | //td[@data-ng-bind="item.gameWeek"]/text()\
#                           | //td[@data-ng-bind="item.opponent"]/text() | //td[@data-ng-bind="item.points"]/text()\
#                           | //td[@data-ng-bind="item.goals"]/text() | //td[@data-ng-bind="item.assists"]/text()\
#                           | //td[@data-ng-bind="item.minutes"]/text() | //td[@data-ng-bind="item.bps"]/text()\
#                           | //td[@data-ng-bind="item.goalsConceded"]/text() | //td[@data-ng-bind="item.influence"]/text() \
#                           | //td[@data-ng-bind="item.creativity"]/text() | //td[@data-ng-bind="item.threat"]/text() \
#                           | //td[@data-ng-bind="item.ictIndex"]/text() | //td[@data-ng-bind="item.selectedBy"]/text() \
#                           | //td[@data-ng-bind="item.netTransfers"]/text()  | //td[@data-ng-bind="item.reds"]/text()\
#                           | //td[@data-ng-bind="item.yellows"]/text() | //td[@data-ng-bind="item.saves"]/text()\
#                           | //td[@data-ng-bind="item.bonus"]/text() | //td[@data-ng-bind="item.cleanSheets"]/text()')

