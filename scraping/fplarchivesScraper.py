from selenium import webdriver
from bs4 import BeautifulSoup
import time
from lxml import html
import unicodecsv as csv

driver = webdriver.Firefox()
driver.implicitly_wait(10)
with open('../data/2016_2017_season_data.csv','a') as csvfile:
    w = csv.writer(csvfile, delimiter=' ')
    for i in range(1,684):
        url = 'http://fplarchives.com/Seasons/1/Players/' + str(i)
        driver.get(url)
        time.sleep(4)
        tree = html.fromstring(driver.page_source)

        notfound = tree.xpath('//div[@data-ng-bind="notFoundMessage"]/text()')
        if notfound:
            break

        line = tree.xpath('//h3[@data-ng-bind="title"]/text() | //p[@data-ng-bind="position"]/text()\
                           | //p[@data-ng-bind="team"]/text() | //td[@data-ng-bind="item.gameWeek"]/text()\
                           | //td[@data-ng-bind="item.opponent"]/text() | //td[@data-ng-bind="item.points"]/text()\
                           | //td[@data-ng-bind="item.goals"]/text() | //td[@data-ng-bind="item.assists"]/text()\
                           | //td[@data-ng-bind="item.minutes"]/text() | //td[@data-ng-bind="item.bps"]/text()\
                           | //td[@data-ng-bind="item.goalsConceded"]/text() | //td[@data-ng-bind="item.influence"]/text() \
                           | //td[@data-ng-bind="item.creativity"]/text() | //td[@data-ng-bind="item.threat"]/text() \
                           | //td[@data-ng-bind="item.ictIndex"]/text() | //td[@data-ng-bind="item.selectedBy"]/text() \
                           | //td[@data-ng-bind="item.netTransfers"]/text()  | //td[@data-ng-bind="item.reds"]/text()\
                           | //td[@data-ng-bind="item.yellows"]/text() | //td[@data-ng-bind="item.saves"]/text()\
                           | //td[@data-ng-bind="item.bonus"]/text() | //td[@data-ng-bind="item.cleanSheets"]/text()')

        w.writerow(line)
