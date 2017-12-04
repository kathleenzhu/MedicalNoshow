#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:30:58 2017

@author: kathleenzhu
"""
#code taken from Nathan Yau's blog post, link given below: 
#https://flowingdata.com/2007/07/09/grabbing-weather-underground-data-with-beautifulsoup/

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup

# Create/open a file called wunder.txt (which will be a comma-delimited file)
f = open('wunder-data.txt', 'w')
 
# Iterate through year, month, and day
for y in range(2014, 2016):
  for m in range(1, 13):
    for d in range(1, 32):
     
      http = urllib3.PoolManager()
    #SBCZ = Vitoria Aeroporto, Brazil the weather station in Vitoria, Brazil
      # Open wunderground.com url
      url = "http://www.wunderground.com/history/airport/SBVT/"+str(y)+ "/" + str(m) + "/" + str(d) + "/DailyHistory.html"
      response = http.request('GET', url)
 
      # Get temperature from page
      soup = BeautifulSoup(response.data,'html.parser')
      #print(soup.prettify())
      #dayTemp = soup.body.nobr.b.string
      dayTemp = soup.find_all(attrs={"class":"wx-data"})[0].span.string #mean temperature
      dayPrecip = soup.find_all(attrs={"class":"wx-data"})[8].span.string #inches of precip
      #print(dayTemp)
      #print(dayPrecip)
      # Format month for timestamp
      if len(str(m)) < 2:
        mStamp = '0' + str(m)
      else:
        mStamp = str(m)
 
      # Format day for timestamp
      if len(str(d)) < 2:
        dStamp = '0' + str(d)
      else:
        dStamp = str(d)
 
      # Build timestamp
      timestamp = str(y) + '-' + mStamp + '-' +  dStamp
 
      # Write timestamp and temperature to file
      f.write(timestamp + ',' + dayTemp + ',' + dayPrecip + '\n')
 
# Done getting data! Close file.
f.close()

