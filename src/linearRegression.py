import preprocess 
import evaluationMetrics

import scipy.stats
import numpy as np
import pandas as pd
import os
import time
import scipy.stats as sts


class LinearClassifier(object):
    """Bayesian Linear Regression
        Note:
            Initialization
        Args:
            data: Training data - 2D array Image
            labels: Training label - 1D array and numberical data
            bClassPrior: Using uniform distribution (TRUE) or prior distribution from dataset (FALSE)
            bRmUseless: Reduce dimension by variances or not
            iRmThreshold: Threshold for variances in dimension reduction
            Likelihood: $p(t|w, X, σ^2) = N(Xw,σ^2I)$, with $σ^2$ is variance of noise (prior) 
    """
    # initialize class variable
    def __init__(self, data,labels,weightVar=1,noiseVar=1,bMulticlass=False,bRmUseless=True):
        #basic inf
        self.LabelIDs = list(set(labels))
        self.nClass = len(self.LabelIDs)
        self.nSample = len(data)
        self.nDim = len(np.array(data[0]).flatten())+1
        self.multiClassifier = []
        
        #prior
        ## input is muW and covW 
        if(type(weightVar) in (list,tuple) ):
            self.weightPrior = weightVar
        else:
            self.weightPrior = [np.zeros([1,self.nDim]),np.diag(np.ones(self.nDim))*weightVar]
        # compute invert in advance
        if (len(self.weightPrior) == 2):
            self.weightPrior.append(np.linalg.pinv(self.weightPrior[1]))
        self.noisePrior = noiseVar # p(\epsilon|X,Y,w)
        #training
        if (bMulticlass):
            self.train_multiclassifier(data,labels)
        else:
            self.train(data,np.array(labels)[np.newaxis].T)
    
    def addBias(self,X,size=1):
        if (size==1):
            return np.concatenate(([1],X),axis=0)
        else:
            bias = np.ones([size,1])
            return np.concatenate((bias,X),axis=1)
    
    def train(self,X,Y):
        X = self.addBias(X,self.nSample)
        self.covW = np.linalg.inv(np.dot(X.T,X)/self.noisePrior+self.weightPrior[2])
        self.muW = (np.dot(X.T,Y)/self.noisePrior+np.dot(self.weightPrior[2],self.weightPrior[0].T))
        self.muW = np.dot(self.covW,self.muW)
    
    def predict(self,Xnew): 
        if (len(np.shape(Xnew))==1):
            Xnew = self.addBias(Xnew,1)
            return [np.dot(Xnew,self.muW),self.noisePrior+np.dot(np.dot(Xnew,self.covW),Xnew[np.newaxis].T)]
        else:
            Xnew = self.addBias(Xnew,len(Xnew))
            return [np.dot(Xnew,self.muW).T,self.noisePrior+np.sum(np.dot(Xnew,self.covW)*Xnew,axis=1)]
    
    def predictMulti(self,Xnew):
        preY = []
        for i in self.LabelIDs:
            preY.append(self.multiClassifier[i].predict(Xnew)[0][0])
        return np.array(preY)
    
    def testing(self,Xnew,Ynew):
        if (len(self.multiClassifier)!=0):
            preY = []
            for i in self.LabelIDs:
                preY.append(self.multiClassifier[i].predict(Xnew)[0][0])
            preY = [np.argmax(np.array(preY),axis=0)]
        else:
            muXnew,varXnew = self.predict(Xnew)
            preY = np.around(muXnew)
        return preY
    
    def train_multiclassifier(self,X,Y):
        for i in self.LabelIDs:
            model = LinearClassifier(X,(Y==i).astype(int),self.weightPrior,self.noisePrior)
            self.multiClassifier.append(model)
    
   
class LogisticRegression(object):
    # initialize class variable
    def __init__(self,
                 data,
                 labels,
                 MHcov=1e-4,
                 weightVar=10,
                 maxIter=1000,
                 bMulticlass=False,
                 bRmUseless=True):
        #basic inf
        self.LabelIDs = list(set(labels))
        self.nClass = len(self.LabelIDs)
        self.nSample = len(data)
        self.nDim = len(np.array(data[0]).flatten()) + 1
        self.multiClassifier = []

        self.maxIter = maxIter
        self.weightlist = []
        self.weightVar = weightVar
        #prior COVARIANCE - INVERSE COVAR - DETERMINANT
        ## input is muW and covW
        if (type(MHcov) in (list, tuple)):
            self.MHcov = MHcov
        else:
            self.MHcov = np.diag(np.ones(self.nDim)) * MHcov

        if (bMulticlass == False):
            self.train(data, labels)
        else:
            self.train_multiclassifier(data, labels)

    def addBias(self, X, size=1):
        bias = np.ones([size, 1])
        return np.concatenate((bias, X), axis=1)

    def sigmoid(self, X):
        return 1. / (1 + np.exp(-X))

    def laplaceComp(self, w, X, Y):
        logg = -(1 / (2 * self.weightVar)) * np.sum(w * w)
        P = self.sigmoid(np.dot(X.astype(np.longdouble), w))
        logl = np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
        logg = logg + logl
        return logg

    def train(self, X, Y):
        X = self.addBias(X, self.nSample)
        wOld = np.zeros(self.nDim)
        for i in range(self.maxIter):
            wNew = np.random.multivariate_normal(wOld, self.MHcov)
            #logPrior = sts.multivariate_normal.logpdf(wOld,mean=wNew,cov=self.MHcov) - \
            #            sts.multivariate_normal.logpdf(wNew,mean=wOld,cov=self.MHcov)
            logLlh = self.laplaceComp(wNew, X, Y) 
            logLlh_ = self.laplaceComp(wOld, X, Y)
            r = logLlh - logLlh_
            if (r >= 0 or np.log(np.random.rand(1)) < r):
                self.weightlist.append(wNew)
                wOld = wNew
        if (len(self.weightlist) > 10):
            self.weightlist = self.weightlist[10:]

    def predict(self, Xnew):
        Xnew = self.addBias(Xnew, len(Xnew))
        return 1 / len(self.weightlist) * np.sum(
            self.sigmoid(np.dot(Xnew, np.array(self.weightlist).T)), axis=1)

    def train_multiclassifier(self, X, Y):
        for i in self.LabelIDs:
            print('Training for class %d' % (i), '---------')
            model = LogisticRegression(
                X, (Y == i).astype(int),
                self.MHcov,
                self.weightVar,
                maxIter=self.maxIter)
            self.multiClassifier.append(model)

    def predictMulti(self, Xnew):
        preY = []
        for i in self.LabelIDs:
            preY.append(self.multiClassifier[i].predict(Xnew))
        return np.array(preY)

    def testing(self, Xnew, Ynew):
        if (len(self.multiClassifier) != 0):
            preY = []
            for i in self.LabelIDs:
                preY.append(self.multiClassifier[i].predict(Xnew)[0][0])
            preY = [np.argmax(np.array(preY), axis=0)]
        else:
            muXnew, varXnew = self.predict(Xnew)
            preY = np.around(muXnew)
        return preY