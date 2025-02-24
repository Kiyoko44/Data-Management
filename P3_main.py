#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.options.mode.chained_assignment = None
import math
import pickle
from datetime import timedelta
import numpy as np
from numpy.fft import fft
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.stats import entropy

dts = '%m/%d/%Y %H:%M:%S'
BWZ = 'BWZ Carb Input (grams)'

# read data from csv files with date and time stamps
def readData(insulin, cgm, dateTimeStamp):
    insulinDatas = pd.read_csv(insulin, low_memory = False)
    insulinD = insulinDatas[['Date', 'Time', BWZ]]
    insulinD['dateTime'] = pd.to_datetime(insulinD['Date'] + ' ' + insulinD['Time'])
    cgmDatas = pd.read_csv(cgm, low_memory = False)
    cgmD = cgmDatas[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
    cgmD.dropna(inplace = True)
    cgmD['dateTime'] = pd.to_datetime(cgmD['Date'] + ' ' + cgmD['Time'])
    return insulinD, cgmD
    
def getMealStart(insulinData):
    carb = insulinData[insulinData[BWZ].notna() & insulinData[BWZ] != 0]
    carb.rename({'dateTime' : 'mealStartTime'}, axis = 1, inplace = True)
    carb = carb[['mealStartTime', BWZ]]
    carb.sort_values(by = 'mealStartTime', inplace = True)
    mealStart = [tuple(c) for c in carb.to_numpy()]
    return mealStart

def getMealData(mealStart):
    validMeal = []
    for m in range(len(mealStart)):
        tStamp, meal = mealStart[m]
        if 0 < m:
            first = mealStart[m-1][0]
            if first > tStamp - timedelta(minutes=30):
                continue
        if m < len(mealStart)-1:
            last = mealStart[m+1][0]
            if last < tStamp + timedelta(minutes=120):
                continue
        validMeal.append((tStamp, meal))
    return validMeal

def getCarbData(cgmData, validCarb):
    sStamp = []
    vCarb = []
    for start, c in validCarb:
        first = start - timedelta(minutes= 30)
        end = start + timedelta(hours=2)
        sets = cgmData[(cgmData['dateTime'] >= start) & (cgmData['dateTime'] <= end)]
        if 0 < len(sets):
            vCarb.append(c)
            sStamp.append(list(sets['Sensor Glucose (mg/dL)'].values))
    return sStamp, vCarb

def Slope(row):
    slopes = []
    for r in range(len(row)-2):
        slopes.append((row[r]+row[r+2]-2*row[r+1])/((r+2-r)*5.0))
    loc = np.where(np.diff(np.sign(slopes)))[0]
    slope = [(s, abs(slopes[s+1]-slopes[s])) for s in loc]
    slope.sort(key = lambda x: x[1], reverse = True)
    return slope[:3]

def freqIndicies(row):
    freq = fft(row)
    top = np.argsort(freq)[::-1][1:4]
    return top.tolist()

def getAttributes(dSet):
    cgmDif = []
    cgmTimeDif = []
    slope1 = []
    slope2 = []
    slope3 = []
    sAttributes = [slope1, slope2, slope3]
    slopeLoc1 = []
    slopeLoc2 = []
    slopeLoc3 = []
    sAttributesLoc = [slopeLoc1, slopeLoc2, slopeLoc3]
    fft1 = []
    fft2 = []
    fft3 = []
    fftAttributes = [fft1, fft2, fft3]
    for d in dSet:
        dMin = min(d)
        dMax = max(d)
        cgmDif.append(dMax-dMin)
        sTuple = Slope(d)
        for r in range(3):
            if r < len(sTuple):
                sAttributes[r].append(sTuple[r][1])
                sAttributesLoc[r].append(sTuple[r][0])
            else:
                sAttributes[r].append(None)
                sAttributesLoc[r].append(None)
        cgmTimeDif.append((d.index(dMax)-d.index(dMin))*5)
        top = freqIndicies(d)
        for r in range(3):
            if r < len(top):
                fftAttributes[r].append(top[r])
            else:
                fftAttributes[r].append(None)
    outcome = pd.DataFrame()
    outcome['cgmDif'] = cgmDif
    outcome['cgmTimeDif'] = cgmTimeDif
    outcome['slope1'] = slope1 
    outcome['slope2'] = slope2 
    outcome['slope3'] = slope3
    outcome['fft1'] = fft1 
    outcome['fft2'] = fft2 
    outcome['fft3'] = fft3
    return outcome

def getGroundTruth(validCarb, binSize=20):
    gTruth = []
    cMin = min(validCarb)
    cMax = max(validCarb)
    for c in validCarb:
        vc = int((c-cMin)/binSize*1.0)
        gTruth.append(vc)
    count = math.ceil((cMax-cMin)-(binSize*1.0))
    return gTruth, int(count)
 
def getGroundTruthAtts(insulin, cgm, dateTimeStamp=dts):
    insulinData, cgmData = readData(insulin, cgm, dateTimeStamp)
    mealStart = getMealStart(insulinData)
    mealData = getMealData(mealStart)
    mealAtts, carbAtts = getCarbData(cgmData, mealData)  
    mealNorm = getNorm(mealAtts)
    mealNorm, carbAtts = clean(mealNorm, carbAtts)
    groundTruth, gtCount = getGroundTruth(carbAtts)
    return mealNorm, groundTruth, gtCount

def norm(n):
    normalize = n-n.min()/((n.max()-n.min())*1.0)
    return normalize

def getNorm(sStamp):
    nCarb = getAttributes(sStamp)
    nCarb = norm(nCarb)
    return nCarb

def clean(nCarb, vCarb):
    nCarb['Carb Input'] = vCarb
    nCarb.dropna(inplace = True)
    vCarb = nCarb['Carb Input']
    nCarb.drop(columns = ['Carb Input'], axis = 1, inplace = True)
    return nCarb, vCarb

def makeMatrix(X, cLabels, gTruth):
    cCount = len(np.unique(cLabels))
    bCount = (len(np.unique(gTruth)))
    gtMatrix = np.zeros((cCount, bCount))
    for m in range(len(X)):
        temp = X[m]
        cIdx = cLabels[m]
        bIdx = gTruth[m]
        gtMatrix[cIdx][bIdx]+=1
    return gtMatrix

def calcEntropyPurity(gtMatrix, X):
    cCount = len(gtMatrix)
    bCount = len(gtMatrix[0])
    eTotal = 0
    pTotal = 0
    for e in range(cCount):
        temp = sum(gtMatrix[e])
        for p in range(bCount):
            gtMatrix[e][p]/=(temp*1.0)
        ent = entropy(gtMatrix[e])
        eTotal += temp+ent
        pur = max(gtMatrix[e])
        pTotal += temp+pur
    eTotal /= len(X)+1.0
    pTotal /= len(X)+1.0
    return eTotal, pTotal

def getEntropyPurity(X, cLabels, groundTruth):
    gtMatrix = makeMatrix(X, cLabels, groundTruth)
    ent, pur = calcEntropyPurity(gtMatrix, X)
    return ent, pur

def getKMeans(X, cCount, groundTruth):
    kmSSE = 0
    km = KMeans(n_clusters=cCount, random_state=2).fit(X)
    kLabels = km.labels_
    kCenter = km.cluster_centers_
    for k in range(len(X)):
        temp = X[k]
        idx = kLabels[k]
        centroid = kCenter[idx]
        d = distance.euclidean(temp, centroid)
        kmSSE = kmSSE + math.pow(d, 2)
    kmEntropy, kmPurity = getEntropyPurity(X, kLabels, groundTruth)
    return kmSSE, kmEntropy, kmPurity

def getCluster(dArray, label):
    cluster = []
    arr = pd.DataFrame(dArray)
    arr['label'] = label
    uniq = arr['label'].unique()
    for u in uniq:
        cluster.append(arr[arr['label'] == u].loc[:, arr.columns != 'label'].to_numpy())
    return cluster

def calcCluster(cList):
    SSE = []
    for c in cList:
        cAvg = sum(c)/(len(c)*1.0)
        temp = 0
        for p in c:
            temp = temp + math.pow(distance.euclidean(p, cAvg),2)
        SSE.append(temp)
    return SSE

def bisectingKMeans(dArray):
    cluster = KMeans(n_clusters = 2, random_state=0).fit(dArray)
    cList = getCluster(dArray, cluster.labels_)
    return cList

def dbscanKMeans(X, groundTruth):
    dArray = []
    cLabels = []
    gtBin = []
    bis = pd.DataFrame(X)
    bis['Ground Truth Bin'] = groundTruth
    xGT = bis.to_numpy()
    dbs = DBSCAN(eps=0.8, min_samples=14).fit(X)
    dbsLabel = dbs.labels_
    cWith = getCluster(xGT, dbsLabel)
    cWithout = [c[:, :-1] for c in cWith]
    SSE = calcCluster(cWithout)
    while N > len(cWith):
        sseMax = np.argsort(SSE)[-1]
        i = bisectingKMeans(cWith[sseMax])
        cWith = cWith[:sseMax]+cWith[sseMax+1:]+i
        cWithout = [c[:, :-1] for c in cWith]
        SSE = calcCluster(cWithout)
    sseDBS = sum(SSE)
    for g in range(len(cWith)):
        cWithData = cWith[g]
        for t in cWithData:
            cLabels.append(g)
            dArray.append(t[:-1])
            gtBin.append(int(t[-1]))
    entropyDBS, purityDBS = getEntropyPurity(dArray, cLabels, gtBin)
    return sseDBS, entropyDBS, purityDBS

atts, groundTruth, gtBin = getGroundTruthAtts('InsulinData.csv', 'CGMData.csv', dts)
X = atts.to_numpy()
N = int(gtBin)
kmSSE, kmEntropy, kmPurity = getKMeans(X, N, groundTruth)
sseDBS, entropyDBS, purityDBS = dbscanKMeans(X, groundTruth)
outcome = pd.DataFrame([[kmSSE, sseDBS, kmEntropy, entropyDBS, kmPurity, purityDBS]])
outcome.to_csv('Result.csv', index=False, header=False)

print("HI")


# In[ ]:




