#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:53:11 2017

@author: kathleenzhu
"""
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
pd.options.mode.chained_assignment = None  # default='warn'
os.chdir('/Users/kathleenzhu/Documents/GitHub/MedicalNoshow')
data = pd.read_csv('No-show-Issue-Comma-300k.csv')
data.columns

#set datetime objects; categorical data; specify factors and bools
data['AppointmentRegistration'] = data['AppointmentRegistration'].astype('datetime64[ns]')
data['AppointmentDate'] = data['ApointmentData'].astype('datetime64[ns]')
data['Gender'] = data['Gender'].astype('category') 
data['DayOfTheWeek'] = data['DayOfTheWeek'].astype('category')
data['DayOfTheWeek'] = pd.Categorical(data['DayOfTheWeek'], ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                    ordered=True)
#data['Status'] = data['Status'].astype('bool')
#data['Diabetes'] = data['Diabetes'].astype('bool')
#data['Alchoholism'] = data['Alcoolism'].astype('bool')
#data['Hypertension'] = data['HiperTension'].astype('bool')
#data['Handicap'] = data['Handcap'].astype('bool')
#data['Smokes'] = data['Smokes'].astype('bool')
#data['Scholarship'] = data['Scholarship'].astype('bool')
#data['Tuberculosis'] = data['Tuberculosis'].astype('bool')
#data['Sms_Reminder'] = data['Sms_Reminder'].astype('bool')


data['Alcoholism'] = data['Alcoolism']
data['Hypertension'] = data['HiperTension']
data['Handicap'] = data['Handcap']

data = data[['Status','AppointmentRegistration','AppointmentDate','AwaitingTime',
             'DayOfTheWeek','Age','Gender','Diabetes','Alcoholism','Hypertension',
             'Handicap','Smokes','Scholarship','Tuberculosis','Sms_Reminder']]
data.columns
data.head()
data.info()
data.describe()

#clean data
    #take out duplicate rows
data_nodup = data.drop_duplicates()
data.shape[0]-data_nodup.shape[0] #drop 343 rows
data = data_nodup

#missing data? NAs?
data.isnull().values.any() #no NA values

#check age, restrict to be non-negative
data['Age'].describe()
data.shape[0] - data[data['Age']>= 0].shape[0] #drop 6 rows
data = data[data['Age']>= 0]

#awaiting time, difference between time appointment was made, and time of actual appointment 
#change to be positive values, because it's a count of the number of days between, easier to handle
data['AwaitingTime'] = data['AwaitingTime']*-1 


#track date of appointment 
data['ApptDate'] = data['AppointmentDate'].apply(lambda x: x.date())

#track month of appointment 
data['ApptMonth'] = data['AppointmentDate'].apply(lambda x: x.month)

#track date of registration 
data['RegDate'] = data['AppointmentRegistration'].apply(lambda x: x.date())

#change Status to a numeric binary varaiable: 0 = No-Show, 1 = Show-Up
data['Status'].unique()
data['Status'][data['Status'] == 'Show-Up'] = 1
data['Status'][data['Status'] == 'No-Show'] = 0
data['Status'] = data['Status'].astype('int64')

#create a dummy variable: 1 = No-Show, 0 = Show-Up
data['NoShow'] = data['Status']
data['NoShow'][data['Status'] == 1] = 0
data['NoShow'][data['Status'] == 0] = 1

#set bool values greater than 1 to 1 for Handicap and Sms_Reminder
data['Sms_Reminder'][data['Sms_Reminder']>1] = 1 #changed 799 
data['Handicap'] [data['Handicap']>1] = 1 #changed 495 

#describe data, high level
data.describe()

#percent no-shows
data['NoShow'].value_counts()/len(data['NoShow'])

#visualize data
    #plot proportion of no-shows by appointment date
bydate = data.groupby('ApptDate')['NoShow'].mean().reset_index() #mean is the proportion of No Shows 
plt.plot_date(bydate['ApptDate'],bydate['NoShow'])  
    #plot proportion of no-shows by registration date
bydate2 = data.groupby('RegDate')['NoShow'].mean().reset_index()
plt.plot_date(bydate2['RegDate'],bydate2['NoShow'])
    
    #Day of the Week
byDOTW = data.groupby('DayOfTheWeek')['NoShow'].count().reset_index()
byDOTW = byDOTW.sort_values('DayOfTheWeek')
y_pos = np.arange(len(byDOTW['NoShow']))
plt.bar(y_pos, byDOTW['NoShow'])

    #Month
byMonth = data.groupby('ApptMonth')['NoShow'].count().reset_index()
byMonth = byMonth.sort_values('ApptMonth')
y_pos = np.arange(len(byMonth['NoShow']))
plt.scatter(y_pos, byMonth['NoShow'])


    #Gender
byGender = data.groupby('Gender')['NoShow'].count().reset_index()
#make boxplots

    #Age
plt.hist([data['Age'][data['NoShow']==0],data['Age'][data['NoShow']==1]], bins = 35,stacked=True, color = ['b','r'])

#Binary Flags
    #diabetes
    #alcoholism
    #hypertension
    #smokes
    #scholarship
    #tuberculosis
    
#basic model: logistic regression

