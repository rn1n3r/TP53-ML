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
        clf = RandomForestClassifier(n_estimators=10, verbose = 3)
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
    
def clfTestProb(data, top, morph, runs):
    isOrgan = []
   
    
    for i in range (0,runs):
    # take a sample for training, leave the rest for testing (cross-validation)
        data_train, data_test, top_train, top_test, morph_train, morph_test = train_test_split(data,top, morph)
        clf = RandomForestClassifier(n_estimators=10, verbose=2)
       
        clf = clf.fit(data_train, top_train)
        top_class = clf.classes_
        clf_predict_top = clf.predict_proba(data_test)
        
        clf = clf.fit(data_train, morph_train)
        clf_predict_morph = clf.predict_proba(data_test)
        morph_class = clf.classes_
    strTop = []
    strMorph = []
    
    
    for prob in clf_predict_top:
        i = prob.tolist().index(max(prob))
        strTop.append( '{:.2f}'.format(max(prob)) + " " + str(top_class[i]))
    for prob in clf_predict_morph:
        i = prob.tolist().index(max(prob))
        strMorph.append( '{:20.2f}'.format(max(prob)) + " " + str(morph_class[i]))
        
    for i,x in enumerate(strTop):
        print(x + " " + strMorph[i])
        
def svmTest(data, top, organ):
    isOrgan = []    
    for x in top:
        if (x == organ):
            isOrgan.append(True)
        else:
            isOrgan.append(False)    
    data_train, data_test, top_train, top_test, organ_train, organ_test = train_test_split(data,top,isOrgan)
    
    svc = svm.SVC(kernel='linear', verbose=2)
    svc.fit(data_train,organ_train)
    
    svc_predict = svc.predict(data_test)
    print(numpy.sum(svc_predict == organ_test)/len(organ_test))

def getData(filepath, germOrSomatic):
    with open(filepath) as f:
        next(f) # skip headers
    
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
        
        if (germOrSomatic == 'somatic'):
            # data fields    
            mut_type= [hash(item[7]) for item in data] # for somatic
            top = [item[30] for item in data]
            morph = [item[34] for item in data]
            loc = [item[2] for item in data]
            sex = [item[45] for item in data]
        else:
            mut_type = [hash(item[15]) for item in data]
            top = [item[39] for item in data]
            morph = [item[41] for item in data]
            loc = [item[10] for item in data]
            sex = [item[32] for item in data]
    # mapping the types
    for i,x in enumerate(sex):
        if (x == 'M'):
            sex[i] = 0
        else:
            sex[i] = 1
    # combine data fields
    data = [list(a) for a in zip(mut_type, loc, sex)]
    return (data, top, morph)
    
    
    
def main():
#    with open('C:\\Users\\Edward\\Documents\\Files\\hack\\biohacks\\topo.csv') as f:
#        reader = csv.reader(f, delimiter = ",")
#        all = [item[1] for item in list(reader)]
    
    data, top, morph= getData('C:\\Users\\Edward\\Documents\\Files\\hack\\biohacks\\germlineData.txt', 'germ')
    
    
    #svmTest(data, top, 'BREAST')
    clfTestProb(data, top, morph, 1)
    #clfTestProb(data, top, 1)
    
main()
    
    

    
