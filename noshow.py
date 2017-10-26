#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:53:11 2017

@author: kathleenzhu
"""
import os
import pandas as pd
import numpy as np 
os.chdir('/Users/kathleenzhu/Documents/GitHub/MedicalNoshow')
data = pd.read_csv('No-show-Issue-Comma-300k-corrected.csv')
data.columns

#set categorical data; specify factors and bools
data['Gender'] = data['Gender'].astype('category') 
data['DayOfTheWeek'] = data['DayOfTheWeek'].astype('category')
data['Status'] = data['Status'].astyper('bool')
data['Diabetes'] = data['Diabetes'].astype('bool')
data['Alchoholism'] = data['Alchoholism'].astype('bool')
data['Hypertension'] = data['Hypertension'].astype('bool')
data['Handicap'] = data['Handicap'].astype('bool')
data['Smokes'] = data['Smokes'].astype('bool')
data['Scholarship'] = data['Scholarship'].astype('bool')
data['Tuberculosis'] = data['Tuberculosis'].astype('bool')
data['Sms_Reminder'] = data['Sms_Reminder'].astype('bool')
 
#describe data, high level
data.info()
data.describe()
