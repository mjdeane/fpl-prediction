import requests
import json
import csv
import argparse
import numpy as np
import os

FPL_URL = "https://fantasy.premierleague.com/api/"
PLAYER_SUMMARY_SUBURL = "element-summary/"
PLAYER_SUMMARY_URL = FPL_URL + PLAYER_SUMMARY_SUBURL
PLAYERS_INFO_SUBURL = "bootstrap-static/"
PLAYERS_INFO_URL = FPL_URL + PLAYERS_INFO_SUBURL

dir_path = os.path.dirname(os.path.realpath(__file__))

def getNextGameweek():
    r = requests.get(PLAYERS_INFO_URL)
    info = r.json()
    events = info["events"]
    gw_num = 0
    for event in events:
        if event["is_next"] == True:
            gw_num = event["id"]
    return gw_num

def getPreviousGameweek():
    r = requests.get(PLAYERS_INFO_URL)
    info = r.json()
    events = info["events"]
    gw_num = 0
    for event in events:
        if event["is_previous"] == True:
            gw_num = event["id"]
    return gw_num

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

def getPlayerInputVectorRow(info, i):
    vector = np.empty(17);
    vector[0] = info['history'][i]['total_points']
    vector[1] = info['history'][i]['goals_scored']
    vector[2] = info['history'][i]['assists']
    vector[3] = info['history'][i]['minutes']
    vector[4] = info['history'][i]['bps']
    vector[5] = info['history'][i]['goals_conceded']
    vector[6] = info['history'][i]['influence']
    vector[7] = info['history'][i]['creativity']
    vector[8] = info['history'][i]['threat']
    vector[9] = info['history'][i]['ict_index']
    vector[10] = info['history'][i]['selected']
    vector[11] = info['history'][i]['transfers_balance']
    vector[12] = info['history'][i]['red_cards']
    vector[13] = info['history'][i]['yellow_cards']
    vector[14] = info['history'][i]['saves']
    vector[15] = info['history'][i]['bonus']
    vector[16] = info['history'][i]['clean_sheets']
    return vector

def getPlayerInputVector(player_id):
    info = getPlayerInfo(player_id)
    gw = getNextGameweek()
    if (len(info['history']) < gw-1):
        return np.zeros(85)
    vector = np.array([])
    for i in range(gw-6,gw-1):
#        print(info['history'][i]['total_points'])
        vector = np.append(vector, getPlayerInputVectorRow(info, i))
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
    with open(os.path.join(os.path.dirname(dir_path),'data','player_input_vectors_')+str(getNextGameweek()), 'w') as outfile:
        w = csv.writer(outfile, delimiter=' ')
        w.writerow(ids)
        for z in Z:
            w.writerow(z)

def getAllPlayerInputVectors():
    if not os.path.isfile(os.path.join(os.path.dirname(dir_path),'data','player_input_vectors_')+str(getNextGameweek())):
        writeAllPlayerInputVectors()
    with open(os.path.join(os.path.dirname(dir_path),'data','player_input_vectors_')+str(getNextGameweek()),'r') as infile:
        r = csv.reader(infile, delimiter=' ')
        l =  list(r)
        ids = np.array(l[0]).astype(float)
        Z = np.array(l[1:]).astype(float)
        return(Z,ids)

def writeSeasonData():
    gw = getNextGameweek()
    info = getAllPlayersInfo()
    Z = np.empty((len(info),17*(gw-1)))
    
    j=0
    for p in info:
        player_info = getPlayerInfo(int(p['id']))
        if (len(player_info['history']) < gw-1):
            continue
        tmpvec = []
        for i in range(0,gw-1):
            tmpvec = np.append(tmpvec,getPlayerInputVectorRow(player_info,i))
        Z[j] = tmpvec
        j += 1
    print(Z.shape)
    Z = np.resize(Z,(j,17*(gw-1)))
    print(Z.shape)
    print(len(info))
    print(j)
    with open(os.path.join(os.path.dirname(dir_path),'data','season_data.csv'), 'w') as outfile:
        w = csv.writer(outfile, delimiter=' ')
        for z in Z:
            w.writerow(z)


def getEvaluationVectors():
    with open(os.path.join(os.path.dirname(dir_path),'data','season_data.csv'), 'r') as infile:
        r = csv.reader(infile, delimiter=' ')
        Z = np.array(list(r))

    gw = getNextGameweek()
    if gw < 5:
        return
    with open(os.path.join(os.path.dirname(dir_path),'data','attackers2_x.csv'), 'w') as out_x:
        wx = csv.writer(out_x, delimiter=' ')
        with open(os.path.join(os.path.dirname(dir_path),'data','attackers2_y.csv'), 'w') as out_y:
            wy = csv.writer(out_y, delimiter=' ')
            for p in Z:
                for i in range(5,gw-2):
                    wx.writerow(p[17*(i-5):17*i])
                    wy.writerow([p[17*i]])

def getIdToNameDict():
    info = getAllPlayersInfo()
    names = dict([])

    for p in info:
        names[p['id']] = p['web_name']

#    print(names.keys())

    return names

#writeSeasonData()
#getEvaluationVectors()
