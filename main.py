#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pandas


# In[2]:


# load data from csv file
import csv
import pandas as pd
import datetime as dt
import time
import math


# In[6]:


mEntries = []
aEntries = []

insulinData = pd.read_csv('InsulinData.csv')
insulinDataSet = insulinData[['Date', 'Time', 'Alarm']]
cgmData = pd.read_csv('CGMData.csv')
cgmDataSet = cgmData[['Date', 'Time', 'Sensor Glucose (mg/dL)']]

insulinData['DateTime'] = pd.to_datetime(insulinData['Date'] + " " + insulinData['Time'], format = '%m/%d/%Y %H:%M:%S')
cgmData['DateTime'] = pd.to_datetime(cgmData['Date'] + " " + cgmData['Time'], format = '%m/%d/%Y %H:%M:%S')

autoStart = insulinData[insulinData['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']['DateTime'].min()

# Split the CGM data by auto and manual

autoCGM = cgmData[cgmData['DateTime'] >= autoStart]
dayAutoCGM = autoCGM[autoCGM['DateTime'].dt.hour >= 6]
nightAutoCGM = autoCGM[autoCGM['DateTime'].dt.hour < 6]

manualCGM = cgmData[cgmData['DateTime'] < autoStart]
dayManualCGM = manualCGM[manualCGM['DateTime'].dt.hour >= 6]
nightManualCGM = manualCGM[manualCGM['DateTime'].dt.hour < 6]

def dateDrop(df, col, tmp = 0, expected = 288):
    cGroup = df.groupby('Date').count()[col]
    keyDrop = list(cGroup[(cGroup/expected) < tmp].keys())
    result = df[~df['Date'].isin(keyDrop)]
    return result

tmp = 0
autos = dateDrop(df = autoCGM, col = 'Sensor Glucose (mg/dL)', tmp = tmp, expected = 288)
autoDay = dateDrop(df = dayAutoCGM, col = 'Sensor Glucose (mg/dL)', tmp = tmp, expected = 216)
autoNight = dateDrop(df = nightAutoCGM, col = 'Sensor Glucose (mg/dL)', tmp = tmp, expected = 72)

manuals = dateDrop(df = manualCGM, col = 'Sensor Glucose (mg/dL)', tmp = tmp, expected = 288)
manualDay = dateDrop(df = dayManualCGM, col = 'Sensor Glucose (mg/dL)', tmp = tmp, expected = 216)
manualNight = dateDrop(df = nightManualCGM, col = 'Sensor Glucose (mg/dL)', tmp = tmp, expected = 72)

def getRange(dataframe, cName, i, expectedDay):
    df = dataframe
    days = len(df['Date'].unique())
    tDays = days * expectedDay
    r = 0
    (rMin, rMax) = i
    if rMin is not None and rMax is not None:
        r = df[(df[cName] >= rMin) & (df[cName] <= rMax)].count()[cName]
    elif rMax is not None:
        r = df[df[cName] < rMax].count()[cName]
    elif rMin is not None:
        r = df[df[cName] > rMin].count()[cName]
    return (r / (tDays * 1.0)) * 100

cName = 'Sensor Glucose (mg/dL)'

# Manual, auto, 
mList = [(manualNight, 288), (manualDay, 288), (manuals, 288)]
aList = [(autoNight, 288), (autoDay, 288), (autos, 288)] 
iList = [(180, None), (250, None), (70, 180), (70, 150), (None, 70), (None, 54)]

for df,expectedDay in mList:
    for i in iList:
        mEntries.append(getRange(df, cName, i, expectedDay))
for df,expectedDay in aList:
    for i in iList:
        aEntries.append(getRange(df, cName, i, expectedDay))

results = pd.DataFrame([mEntries, aEntries])

results.to_csv('Results.csv', index = False, header = False)


# In[ ]:




