# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.datasets
import sklearn.linear_model
import scipy.io
from numpy.linalg import inv
import itertools
import time
import statsmodels.api as sm


dir="/Users/cao/Desktop/DS Spring/ML for DS/hw1/"
Inputdata = scipy.io.loadmat(dir+'wine.mat')
Xtrain = np.array(Inputdata['data'])
Xtrain.shape
Ytrain = np.array(Inputdata['labels'])

res=np.linalg.lstsq(Xtrain,Ytrain)

estimator = np.array(res[0])

Y_pred=np.dot(Xtrain,estimator)
dy = Ytrain-Y_pred

mse = np.mean(dy**2)

#check for normal equation
estimator_ne =np.dot(inv(np.dot(Xtrain.T,Xtrain)),np.dot(Xtrain.T,Ytrain))

# we can observe that the estimator satisfied the normal equation.
Xtest = np.array(Inputdata['testdata'])
Ytest = np.array(Inputdata['testlabels'])

Y_predtest=np.dot(Xtest,estimator)
dyt = Ytest-Y_predtest

mset = np.mean(dyt**2)
feature =np.arange(1,12,1)
#2
def processSubset(feature_set):
    model = sm.OLS(Ytrain,Xtrain[:,list(feature_set)])
    regr = model.fit()
    mse = ((regr.predict(Xtrain[:,list(feature_set)]).reshape(3249,1) - Ytrain) ** 2).mean()
    return {"model":regr, "mse":mse,"Set":list(feature_set)}
# Fit model on feature_set and calculate RSS


def getBest(k):
    results = []
    for combo in itertools.combinations(feature, k):
        combo = [0]+list(combo)
        print(combo)
        results.append(processSubset(combo))
    models = pd.DataFrame(results)
# Choose the model with the highest RSS
    best_model = models.loc[models["mse"].argmin()]
    print("Processed ", models.shape[0], "models on", k)
# Return the best model, along with some other useful information about the model
    return best_model,models
k=3
bmodels = pd.DataFrame(columns=["mse", "model","Set"])
models = pd.DataFrame(columns=["mse", "model","Set"])
bmodels,models = getBest(k)
bmodels["model"].summary()

bestset = bmodels["Set"]
bestest = bmodels["model"].params
Xbtest = Xtest[:,bestset]


bY_pred=np.dot(Xbtest,np.array(bestest).reshape(k+1,1)) #change this to nparray!!!!
bdy = Ytest-bY_pred
bmset = np.mean(bdy**2)

df = pd.DataFrame(data =Xtest)
cor = df.corr()

set1 = set(bestset)
set2 = set(feature)-set1
col = list(set1-set([0]))
row = list(set2)

result = cor.iloc[col,row]


    