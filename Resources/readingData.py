#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the script you can use to read the data provided in directory dataset. Here h1, n1, s1, 
ls1, and lh1 refers to five classes namely:
    h1: hard sweep
    n1: neutral regions
    s1: soft sweep
    ls1: linked to soft sweep
    lh1: linked to hard sweep

"""

import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

numSubWins = 11
trainingDir = '/home/documents/data/training/'
testingDir = '/home/documents/data/testing/'

hard = np.loadtxt(trainingDir+"hard.fvec",skiprows=1)
nDims = int(hard.shape[1] / numSubWins)
h1 = np.reshape(hard,(hard.shape[0],nDims,numSubWins))
neut = np.loadtxt(trainingDir+"neut.fvec",skiprows=1)
n1 = np.reshape(neut,(neut.shape[0],nDims,numSubWins))
soft = np.loadtxt(trainingDir+"soft.fvec",skiprows=1)
s1 = np.reshape(soft,(soft.shape[0],nDims,numSubWins))
lsoft = np.loadtxt(trainingDir+"linkedSoft.fvec",skiprows=1)
ls1 = np.reshape(lsoft,(lsoft.shape[0],nDims,numSubWins))
lhard = np.loadtxt(trainingDir+"linkedHard.fvec",skiprows=1)
lh1 = np.reshape(lhard,(lhard.shape[0],nDims,numSubWins))

both=np.concatenate((h1,n1,s1,ls1,lh1))
y=np.concatenate((np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1))))


both = both.reshape(both.shape[0],nDims,numSubWins,1)
if (trainingDir==testingDir):
    X_train, X_test, y_train, y_test = train_test_split(both, y, test_size=0.2)
else:
    X_train = both
    y_train = y
    #testing data
    hard = np.loadtxt(testingDir+"hard.fvec",skiprows=1)
    h1 = np.reshape(hard,(hard.shape[0],nDims,numSubWins))
    neut = np.loadtxt(testingDir+"neut.fvec",skiprows=1)
    n1 = np.reshape(neut,(neut.shape[0],nDims,numSubWins))
    soft = np.loadtxt(testingDir+"soft.fvec",skiprows=1)
    s1 = np.reshape(soft,(soft.shape[0],nDims,numSubWins))
    lsoft = np.loadtxt(testingDir+"linkedSoft.fvec",skiprows=1)
    ls1 = np.reshape(lsoft,(lsoft.shape[0],nDims,numSubWins))
    lhard = np.loadtxt(testingDir+"linkedHard.fvec",skiprows=1)
    lh1 = np.reshape(lhard,(lhard.shape[0],nDims,numSubWins))

    both2=np.concatenate((h1,n1,s1,ls1,lh1))
    X_test = both2.reshape(both2.shape[0],nDims,numSubWins,1)
    y_test=np.concatenate((np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1))))

Y_train = np_utils.to_categorical(y_train, 5)
Y_test = np_utils.to_categorical(y_test, 5)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.5)        
        
        
#print(X_train.shape)
print("training set has %d examples" % X_train.shape[0])
print("validation set has %d examples" % X_valid.shape[0])
print("test set has %d examples" % X_test.shape[0])       





        
        
        