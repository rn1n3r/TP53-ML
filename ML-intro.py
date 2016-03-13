# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:13:47 2016

@author: Edward
"""

import numpy
from sklearn import datasets, svm, neighbors
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
data = iris.data

labels = iris.target

knn = neighbors.KNeighborsClassifier()

data_train, data_test, labels_train, labels_test = train_test_split(data,labels)

knn.fit(data_train, labels_train)
knn_predict = knn.predict(data_test)

print("k-Nearest Neighbor")
print("Actual: ",labels_test)
print("Prediction: ", knn_predict)

print("Percentage: ", numpy.sum(knn_predict == labels_test)/len(labels_test))

svc = svm.SVC(kernel='linear')
svc.fit(data_train,labels_train)

svc_predict = svc.predict(data_test)
print("SVM: ", svc_predict)
print("Percentage: ", numpy.sum(svc_predict == labels_test)/len(labels_test))