import requests
import json
import csv
import argparse
import numpy as np
import os

FPL_URL = "https://fantasy.premierleague.com/api/"
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