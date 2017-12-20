import csv

data = []

with open('2016_2017_season_data.csv', 'r') as csvread:
    data = list(csv.reader(csvread, delimiter=' '))


#removes players who been registered for less than 6 gameweeks
data = filter(lambda x: int(x[3]) <= 33 ,data)


# sorts players into defender (goalkeeper, defender) and attacker(midfielder, forward) categories
# points are calculated differently, so we will train the model separately on each category
# the subcategories have points calculated slightly differently, but I expect the extra training data to more than make up for any error introduced
# TODO: account for difference between goalkeeper and defender, as well as between midfielder and forward
defenders = filter(lambda x: x[2] == "Goal Keeper" or x[2] == "Defender", data)
attackers = filter(lambda x: x[2] == "Midfielder" or x[2] == "Forward", data)


# remove name, position, and team from each entry
#TODO: take into account team position in table
defenders = map(lambda x: x[3:], defenders)
attackers = map(lambda x: x[3:], attackers)


# remove gameweek numbers and opponents (for now) from data
#TODO: leave opponents in as a number representing their place in the table at that point in the season
def remove_gameweeks(x):
    l = len(x) / 19
    out = []
    for i in range(0,l-1):
        out.append(x[i*19+2:(i+1)*19])
    out.append(x[19*(l-1)+2:])
    return out

defenders = map(remove_gameweeks, defenders)
attackers = map(remove_gameweeks, attackers)


#convert to float lists
defenders = map(lambda x: map(lambda y: map(float, y), x), defenders)
attackers = map(lambda x: map(lambda y: map(float, y), x), attackers)


# partition data into x and y vectors
# currently we use have input vector as the last five results, output vector is score
#TODO: use data from further back to predict score
#TODO: use next opponent('s position in table) as part of input vector
with open('attackers_x.csv', 'w') as x_att:
    with open('attackers_y.csv', 'w') as y_att:
        xw = csv.writer(x_att, delimiter=' ')
        yw = csv.writer(y_att, delimiter=' ')
        for r in attackers:
            for i in range(5,len(r)):
                yw.writerow(r[i][:1])
                xw.writerow([item for sublist in r[i-5:i] for item in sublist])

with open('defenders_x.csv', 'w') as x_def:
    with open('defenders_y.csv', 'w') as y_def:
        xw = csv.writer(x_def, delimiter=' ')
        yw = csv.writer(y_def, delimiter=' ')
        for r in defenders:
            for i in range(5,len(r)):
                yw.writerow(r[i][:1])
                xw.writerow([item for sublist in r[i-5:i] for item in sublist])
