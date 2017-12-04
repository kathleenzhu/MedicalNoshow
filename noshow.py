#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:53:11 2017

@author: kathleenzhu
"""
#import modules
#import os
#os.chdir('/Users/kathleenzhu/Documents/GitHub/MedicalNoshow')
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from patsy import dmatrices
import seaborn as sns #for correlation matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
pd.options.mode.chained_assignment = None  # default='warn', so we don't get chaining warning

#read data
data = pd.read_csv('No-show-Issue-Comma-300k.csv')
data.columns
weather = pd.read_csv('wunder-data.txt', header = None)

#set datetime objects; categorical data; specify factors; correct spelling
data['AppointmentRegistration'] = data['AppointmentRegistration'].astype('datetime64[ns]')
data['AppointmentDate'] = data['ApointmentData'].astype('datetime64[ns]')
data['DayOfTheWeek'] = data['DayOfTheWeek'].astype('category')
data['DayOfTheWeek'] = pd.Categorical(data['DayOfTheWeek'], ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                    ordered=True)
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

#==============================================================================
#
# clean data
#
#==============================================================================
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
data['WaitTime'] = data['AwaitingTime']*-1 


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

#
data['Gender'][data['Gender'] == 'F'] = 1
data['Gender'][data['Gender'] == 'M'] = 0
data['Gender'] = data['Gender'].astype('int64')

#set bool values greater than 1 to 1 for Handicap and Sms_Reminder
data['Sms_Reminder'][data['Sms_Reminder']>1] = 1 #changed 799 
data['Handicap'] [data['Handicap']>1] = 1 #changed 495 

#describe data, high level
data.describe()

#percent no-shows
data['NoShow'].value_counts()/len(data['NoShow'])

#MERGE DATA
data['merge']=data['ApptDate'].astype(str)
data = data.merge(weather, how = 'left', left_on = 'merge', right_on = 0)
data = data.rename(columns = {1:'Temp',2:'Precip'})
data.columns
data2 = data.drop(['merge',0], axis = 1)
#==============================================================================
#
# visualize data
#
#==============================================================================

#Binary Indicators - look at correlation matrix for the varibles
    #diabetes
    #alcoholism
    #hypertension
    #smokes
    #scholarship
    #tuberculosis
binaries = data[['NoShow','Gender','Diabetes','Alcoholism','Hypertension',
             'Handicap','Smokes','Scholarship','Tuberculosis','Sms_Reminder']]

f, ax = plt.subplots(figsize=(10, 8))
corr = binaries.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

    #plot proportion of no-shows by appointment date
bydate = data.groupby('ApptDate')['NoShow'].mean().reset_index() #mean is the proportion of No Shows 
plt.plot_date(bydate['ApptDate'],bydate['NoShow'])  
plt.xlabel('Appointment Date')
plt.ylabel('Proportion No-Show')

    #plot proportion of no-shows by registration date
bydate2 = data.groupby('RegDate')['NoShow'].mean().reset_index()
plt.plot_date(bydate2['RegDate'],bydate2['NoShow'])
plt.xlabel('Registration Date')
plt.ylabel('Proportion No-Show')
#doesn't seem to have a trend over time
  
    #plot proportion of no-shows by wait time
bywait = data.groupby('WaitTime')['NoShow'].mean().reset_index() #mean is the proportion of No Shows 
plt.scatter(bywait['WaitTime'],bywait['NoShow'])  
plt.xlabel('Wait Times')
plt.ylabel('Proportion No-Show')
#longer wait times seems to be associated with higher proportions of NoShows 

    #Day of the Week - count
byDOTW = data.groupby('DayOfTheWeek')['NoShow'].count().reset_index()
byDOTW = byDOTW.sort_values('DayOfTheWeek')
y_pos = np.arange(len(byDOTW['NoShow']))
plt.bar(y_pos, byDOTW['NoShow'])
plt.xticks(y_pos, ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'])
plt.xlabel('Weekday')
plt.ylabel('Proportion No-Show')
#basically no appointments on Sat, Sun, take a look at weekdays

    #Day of the Week - Weekday Proportion
byDOTW = data.groupby('DayOfTheWeek')['NoShow'].mean().reset_index()
byDOTW = byDOTW.sort_values('DayOfTheWeek')
y_pos = np.arange(5)
plt.bar(y_pos, byDOTW['NoShow'][:5])
plt.xticks(y_pos, ['Mon','Tues','Wed','Thurs','Fri'])
plt.xlabel('Weekday')
plt.ylabel('Proportion No-Show')

    #Month
byMonth = data.groupby('ApptMonth')['NoShow'].count().reset_index()
byMonth = byMonth.sort_values('ApptMonth')
y_pos = np.arange(len(byMonth['NoShow']))
plt.bar(y_pos+1, byMonth['NoShow'])
plt.xlabel('Month')
plt.ylabel('Proportion No-Show')


    #Gender
byGender = data.groupby('Gender')['NoShow','Status'].mean().reset_index()
plt.bar(range(2), byGender['NoShow'], color = 'r')
plt.bar(range(2), byGender['Status'], color = 'b', bottom = byGender['NoShow'])
plt.xticks(range(2), ['Female','Male'])
plt.ylabel('Proportion No-Show')


    #Age
plt.hist([data['Age'][data['NoShow']==0],data['Age'][data['NoShow']==1]], bins = 35,stacked=True, color = ['b','r'],label=['Show','NoShow'])
plt.legend()
plt.xlabel('Age')
plt.ylabel('Number of Appointments')

    #Temperature

    #Precipitation 
    
#==============================================================================
#
# create training and development sets 
#
#==============================================================================
#predictors: wait time, age, day of the week, month, scholarship, hypertension, smoke
data.columns
y, X = dmatrices('NoShow ~ WaitTime + Age + DayOfTheWeek +\
                  ApptMonth + Scholarship + Hypertension + Smokes',
                  data, return_type="dataframe")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#==============================================================================
#
# basic model: logistic regression
#
#==============================================================================
#format data
model = LogisticRegression(fit_intercept = False, C = 1e9) 
#we choose a large tuning parameter, C, so we don't consider any regularization in our baseline 
model.fit(X_train, y_train)
model.coef_

#predicted values
predicted = model.predict(X_test)

#evaluation, accuracy
metrics.accuracy_score(y_test, predicted)




