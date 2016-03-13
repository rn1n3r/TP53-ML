# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:09:15 2016

@author: Edward

"""

import csv
import numpy
from sklearn import datasets, svm, neighbors
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

def knnTest(data, top, runs, organ):
    # check if the topography is a certain location or not
    isOrgan = []
    for x in top:
        if (x ==organ):
            isOrgan.append(True)
        else:
            isOrgan.append(False)
    knn = neighbors.KNeighborsClassifier()
    avgVer = 0
    avgPredict = 0
    for i in range (0,runs):
    # take a sample for training, leave the rest for testing (cross-validation)
        data_train, data_test, top_train, top_test, organ_train, organ_test = train_test_split(data,top,isOrgan)
        knn.fit(data_train, organ_train)
        knn_ver = knn.predict(data_test)
        avgVer = avgVer + numpy.sum(knn_ver == organ_test)/len(organ_test)
        
        knn.fit(data_train, top_train)
        knn_predict = knn.predict(data_test)
        avgPredict = avgPredict + numpy.sum(knn_predict == top_test)/len(organ_test)


    avgVer = avgVer / runs
    avgPredict = avgPredict / runs
    print("Percentage Verification " + organ + ": ", avgVer)
    print("Percentage Prediction: ", avgPredict)

def clfTest(data, top, runs, organ):
    isOrgan = []
    breast = 0
    for x in top:
        if (x ==organ):
            isOrgan.append(True)
        else:
            isOrgan.append(False)
    avgVer = 0
    avgPredict = 0
    for i in range (0,runs):
    # take a sample for training, leave the rest for testing (cross-validation)
        data_train, data_test, top_train, top_test, organ_train, organ_test = train_test_split(data,top,isOrgan)
        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(data_train, organ_train)
        clf_ver = clf.predict(data_test)
        clf = clf.fit(data_train, top_train)
        clf_predict = clf.predict(data_test)
        avgVer = avgVer + numpy.sum(clf_ver == organ_test)/len(organ_test)
        avgPredict = avgPredict + numpy.sum(clf_predict == top_test)/len(organ_test)


    avgVer = avgVer / runs
    avgPredict = avgPredict / runs
    print("Percentage Verification " + organ + ": ", avgVer)
    print("Percentage Prediction: ", avgPredict)

    
    

    
with open('C:\\Users\\Edward\\Documents\\Files\\hack\\biohacks\\topo.csv') as f:
    reader = csv.reader(f, delimiter = ",")
    all = [item[1] for item in list(reader)]
    

with open('C:\\Users\\Edward\\Documents\\Files\\hack\\biohacks\\germlineData.txt') as f:
    next(f) # skip headers

    reader = csv.reader(f, delimiter="\t")
    data = list(reader)
    
    # data fields    
    id = [item[0] for item in data]
    #type= [hash(item[7]) for item in data] # for somatic
#    prot = [item[30] for item in data]
#    loc = [item[2] for item in data]
#    sex = [item[45] for item in data]
    typeUnhash = [item[15] for item in data]
    type= [hash(item[15]) for item in data]
    top = [item[41] for item in data]
    loc = [item[10] for item in data]
    sex = [item[32] for item in data]
# mapping the types
for i,x in enumerate(sex):
    if (x == 'M'):
        sex[i] = 0
    else:
        sex[i] = 1
    



# combine data fields
data = [list(a) for a in zip(type, loc, sex)]

clfTest(data, top, 100, 'Sarcoma, NOS')

