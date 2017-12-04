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
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

#MERGE DATA and drop redundant columns
data['merge']=data['ApptDate'].astype(str)
data = data.merge(weather, how = 'left', left_on = 'merge', right_on = 0)
data = data.rename(columns = {1:'Temp',2:'Precip'})
data = data.drop(['Status','AppointmentRegistration','AppointmentDate', 'AwaitingTime',
                   'merge',0], axis = 1)

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
byTemp = data.groupby('Temp')['NoShow'].mean().reset_index()
plt.scatter(byTemp['Temp'],byTemp['NoShow'])  
plt.xlabel('Temperature')
plt.ylabel('Proportion No-Show')

    #Precipitation 
byPercip = data.groupby('Precip')['NoShow'].mean().reset_index()
byPercip = byPercip.iloc[0:25,:] #drop an outlier
plt.scatter(byPercip['Precip'],byPercip['NoShow'])   
plt.xlabel('Precipitation')
plt.ylabel('Proportion No-Show')


#==============================================================================
#
# CHANGE ENCODING FROM 0,1 TO -1,1 
#
#==============================================================================
binaries = data[['NoShow','Gender','Diabetes','Alcoholism','Hypertension',
             'Handicap','Smokes','Scholarship','Tuberculosis','Sms_Reminder']]
for i in binaries:
    data[i][data[i] == 0] = -1
    
#==============================================================================
#
# create training and development sets 
#
#==============================================================================
#predictors: wait time, age, day of the week, month, scholarship, hypertension, smoke
data.columns
y, X = dmatrices('NoShow ~ WaitTime + Age + DayOfTheWeek +\
                  ApptMonth + Scholarship + Hypertension + Precip',
                  data, return_type="dataframe")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#==============================================================================
#
# basic model: least squares regression
#
#==============================================================================

lsregr = linear_model.LinearRegression()
lsregr.fit(X_train,y_train)
predicted = lsregr.predict(X_test)
predicted[predicted >= 0] = 1
predicted[predicted < 0] = -1
metrics.accuracy_score(y_test, predicted)

#==============================================================================
#
# basic model: ridge regression
#
#==============================================================================

ridge = linear_model.Ridge()
ridge.fit(X_train,y_train)
predicted = ridge.predict(X_test)
predicted[predicted >= 0] = 1
predicted[predicted < 0] = -1
metrics.accuracy_score(y_test, predicted)

#==============================================================================
#
# hinge regression
#
#==============================================================================
hinge = linear_model.SGDClassifier(loss = 'hinge',penalty = 'l2', alpha = 0.01,max_iter=5, tol=None) 
hinge.fit(X_train, y_train)
hinge.coef_
predicted = hinge.predict(X_test)
metrics.accuracy_score(y_test, predicted)

#==============================================================================
#
# logistic regression
#
#==============================================================================
#format data
logistic = LogisticRegression(fit_intercept = False, C = 2) 
#we choose a large tuning parameter, C, so we don't consider any regularization in our baseline 
logistic.fit(X_train, y_train)
logistic.coef_

#predicted values
predicted = logistic.predict(X_test)

#evaluation, accuracy
metrics.accuracy_score(y_test, predicted)


#==============================================================================
#
# decision trees
#
#==============================================================================
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=5, min_samples_leaf=2)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)
y_pred
metrics.accuracy_score(y_test,y_pred)*100



clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                     max_depth=5, min_samples_leaf=2)
clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
y_pred
metrics.accuracy_score(y_test,y_pred)*100


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
metrics.accuracy_score(y_test, predicted)








clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], y)
