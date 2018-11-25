#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:19:59 2018

@author: cqj
"""

#HW4 Problem1

import scipy.io
import numpy as np
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
import statsmodels.api as sm

dir="/Users/cqj/Desktop/Columbia/2018Spring/COMS 4721 ML/Homework 4/"
input1 = scipy.io.loadmat(dir+'hw4data.mat')

xdata = np.array(input1['data'])
xdata.shape
ydata = np.array(input1['labels'])

xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.75, random_state=1)

res=np.linalg.lstsq(xtrain,ytrain)
estimator = np.array(res[0])
ypred=np.dot(xtrain,estimator)
diffy = ytrain-ypred
mse = np.mean(diffy**2)

estimator_ne =np.dot(inv(np.dot(xtrain.T,xtrain)),np.dot(xtrain.T,ytrain))

def processSubset(feature_set):
    model = sm.OLS(ytrain,xtrain[:,list(feature_set)])
    regr = model.fit()
    mse = ((regr.predict(xtrain[:,list(feature_set)]).reshape(len(ytrain),1) - ytrain) ** 2).mean()
    return {"model":regr, "mse":mse,"Set":list(feature_set)}


