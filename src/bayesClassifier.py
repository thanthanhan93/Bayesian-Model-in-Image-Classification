from preprocess import *
from evaluationMetrics import *

import scipy.stats
import numpy as np
import pandas as pd
import os
import time

class NaiveBayesClassifier(object):
    """Naive Bayesian Classifer
        Note:
            Initialization
        Args:
            data: Training data - 2D array Image
            labels: Training label - 1D array and numberical data
            bClassPrior: Using uniform distribution (TRUE) or prior distribution from dataset (FALSE)
            bRmUseless: Reduce dimension by variances or not
            iRmThreshold: Threshold for variances in dimension reduction
            smoothing: Coefficient to smooth variances 
    """
    # initialize class variable
    def __init__(self, data,labels,bClassPrior=True,bRmUseless=True,iRmThreshold=1e-2,smoothing=1):
        #basic inf
        self.nClass = len(set(labels))
        self.nSample = len(data)
        self.nDim = len(np.array(data[0]).flatten())       
        #data
        self.data = DataPreprocessing.flattenImage(data)
        self.labels = np.array(labels).astype(int)
        
        # smoothing 
        self.smoothing = smoothing
        # remove useless dimension
        self.del_ind = []
        if (bRmUseless):
            self.removeUselessDimension(iRmThreshold)
            
        # Prior
        if (bClassPrior): 
            #using uniform distribution
            self.classPrior = np.array([1.]*self.nClass)/self.nClass 
        else:   
            #using data distribution
            self.classPrior = self.computeClassPrior()
        
        #Prior p(X|x,y)
        self.componentPrior = self.computeComponentPrior()       
    
    # remove useless dimension which is less than threshold value
    def removeUselessDimension(self, rmThreshold):
        for i in range(self.nDim):
            if (np.var(self.data[:,i]) < rmThreshold):
                self.del_ind.append(i)
        self.data = np.delete(self.data, self.del_ind, axis=1)
        self.nDim = self.nDim - len(self.del_ind)
        
    # compute mean-std for training data
    def computeComponentPrior(self):
        dist = []
        for i in range(self.nClass):
            mean = np.mean(self.data[self.labels==i],axis=0)
            var = np.var(self.data[self.labels==i],axis=0)
            
            mvnNorm = scipy.stats.multivariate_normal(mean,np.diag(var+self.smoothing),allow_singular=True)
            dist.append([mean,var,mvnNorm])
        return np.array(dist)
    
    # compute class prior for training data
    def computeClassPrior(self):
        his = np.bincount(self.labels)
        return his/np.sum(his)
    
    #  compute log p(Xnew|X,labels) by normal pdf in scipy
    def computeLogLikelihood_UnivariateGauss(self,xnew):
        joint_log_likelihood = []
        for classID in range(self.nClass):
            probXnew = np.array([1],dtype=np.longdouble)
            for i in range(self.nDim):
                # obtain mean-variance of the class
                mean = self.componentPrior[classID][0][i]
                var = self.componentPrior[classID][1][i]
                
                # standard deviation != 0 
                if (var != 0):
                    normDist = scipy.stats.norm(mean,np.sqrt(var))
                    probXnew += normDist.logpdf(xnew[i])
                # std = 0 and mean is different => not belong to this class
                elif(mean != xnew[i]):
                    probXnew = -np.inf
                    break
            
            # compute log likelihood
            probClass = self.classPrior[classID]
            probClassXnew = probXnew+np.log(probClass)
            
            # retrive list of likelihood of classes
            joint_log_likelihood.append(probClassXnew)    
        return joint_log_likelihood
    
    # compute log p(Xnew|X,labels) by multivariate normal pdf of scipy
    def computeLogLikelihood_MultivariateGauss(self,xnew):
        joint_log_likelihood = []
        for classID in range(self.nClass):
            mvnNorm = self.componentPrior[classID][2]
            
            # compute log likelihood
            lgllh = mvnNorm.logpdf(xnew)
            lgprior = np.log(self.classPrior[classID])
            
            # retrive list of likelihood of classes
            joint_log_likelihood.append(lgllh+lgprior)    
        return joint_log_likelihood

    # compute log p(Xnew|X,labels) by self-implementation
    def computeLogLikelihood_UnivariateGauss_SelfImple(self,xnew):
        joint_log_likelihood = []
        for i in range(self.nClass):
            # obtain mean-variance of the class
            mean = np.array(self.componentPrior[i][0])
            var = np.array(self.componentPrior[i][1])
            var = var + self.smoothing
            
            # compute log likelihood over all classes
            jointy = np.log(self.classPrior[i])
            jointx = - 0.5 * np.sum(np.log(2. * np.pi * var))
            jointx -= 0.5 * np.sum(((xnew - mean) ** 2) / var)
            joint_log_likelihood.append(jointy + jointx)

        return joint_log_likelihood
          
    # predict class and compute prob of Xnew on each class
    def predict(self,xnew,mode='multivariate'):
        """
        Note:
            Predict on a single test
        Args:
            xnew: a testing vector - 1D array
            mode: method to compute log likelihood (multivariate,univariate,univariateSelf)
        Returns:
            Label of the testing sample and array of log likelihood over class
        """
        probList = []
        #remove useless dimension if possible
        if (len(self.del_ind)!=0):
                xnew= np.delete(xnew,self.del_ind,axis=0)
        
        if (mode == 'multivariate'):
            joint_log_likelihood = self.computeLogLikelihood_MultivariateGauss(xnew)
        elif (mode == 'univariate'):
            joint_log_likelihood = self.computeLogLikelihood_UnivariateGauss(xnew)
        elif(mode =='univariateSelf'):
            joint_log_likelihood = self.computeLogLikelihood_UnivariateGauss_SelfImple(xnew)
        
        return [np.argmax(joint_log_likelihood),joint_log_likelihood]
    
    def testing_Single(self,testing_data,testing_label,mode='multivariate'):
        """
        Note:
            Find test-likelihood and accuracy on test set
        Args:
            testing_data: 2D ndarray
            testing_label: 1D ndarray
            mode: method to compute log likelihood (multivariate,univariate,univariateSelf)
        Returns:
            return accuracy and testlikelihood for test set
        """
        start = time.time()
        preLabels = []
        testing_data = DataPreprocessing.flattenImage(testing_data)
        for i in range(len(testing_data)):
            predictLabel, predictprob = self.predict(testing_data[i],mode=mode)
            preLabels.append(predictLabel)
        return EvaluationMetrics.testingEvaluation(preLabels,testing_label,self.nClass,start,mode,False)
    
    def multiTest_Univariate(self,testing_data):
        joint_log_likelihood = []
        for i in range(self.nClass):
            mean = np.array(self.componentPrior[i][0])
            var = np.array(self.componentPrior[i][1]) + self.smoothing
            
            jointy = np.log(self.classPrior[i])
            jointx = - 0.5 * np.sum(np.log(2. * np.pi * var))
            jointx -= 0.5 * np.sum(((testing_data - mean) ** 2) / var,axis=1)
            joint_log_likelihood.append(jointy + jointx)
            
        joint_log_likelihood = np.array(joint_log_likelihood)
        return [np.argmax(joint_log_likelihood,axis=0),joint_log_likelihood]
    
    def multiTest_Multivariate(self, testing_data):
        joint_log_likelihood = []
        for i in range(self.nClass):
            mvnNorm = self.componentPrior[i][2]
            lgllh = mvnNorm.logpdf(testing_data)
            lgprior = np.log(self.classPrior[i])
            joint_log_likelihood.append(lgllh+lgprior) 
            
        joint_log_likelihood = np.array(joint_log_likelihood)    
        return [np.argmax(joint_log_likelihood,axis=0),joint_log_likelihood]

    def testingOptimized(self,testing_data,testing_label,mode="univariate",bTestLLH=False,bLabel=False):
        """
        Note:
            Find test-likelihood and accuracy on test set
        Args:
            testing_data: 2D ndarray
            testing_label: 1D ndarray
            mode: method to compute log likelihood (multivariate,univariate)
        Returns:
            return accuracy and testlikelihood for test set or the predict probability on each class
        """
        start = time.time()
        testing_data = np.delete(DataPreprocessing.flattenImage(testing_data),self.del_ind,axis=1)
        if (mode == "univariate"):
            preLabels = self.multiTest_Univariate(testing_data)
        elif (mode == "multivariate"):
            preLabels = self.multiTest_Multivariate(testing_data)    
        
        metricsVal = EvaluationMetrics.testingEvaluation(preLabels,testing_label,self.nClass,start,'Batch',bTestLLH)
        if (bLabel==True):
            return preLabels[0]
        return metricsVal
        

class BayesClassifier(object):
    # initialize class variable
    def __init__(self, data,labels,bClassPrior=True,bRmUseless=True,iRmThreshold=1e-2,smoothing=1):
        #basic inf
        self.nClass = len(set(labels))
        self.nSample = len(data)
        self.nDim = len(np.array(data[0]).flatten())       
        #data
        self.data = DataPreprocessing.flattenImage(data)
        self.labels = np.array(labels).astype(int)
        
        # smoothing 
        self.smoothing = smoothing
        # remove useless dimension
        self.del_ind = []
        if (bRmUseless):
            self.removeUselessDimension(iRmThreshold)
            
        # Prior
        if (bClassPrior): 
            #using uniform distribution
            self.classPrior = np.array([1.]*self.nClass)/self.nClass 
        else:   
            #using data distribution
            self.classPrior = self.computeClassPrior()
        
        #Prior p(X|x,y)
        self.componentPrior = self.computeComponentPrior()       
    
    # remove useless dimension which do not help classification
    def removeUselessDimension(self,rmThreshold):
        for i in range(self.nDim):
            if (np.var(self.data[:,i]) < rmThreshold):
                self.del_ind.append(i)
        self.data = np.delete(self.data,self.del_ind,axis=1)
        self.nDim = self.nDim - len(self.del_ind)
    
    # compute class prior for training data
    def computeClassPrior(self):
        his = np.bincount(self.labels)
        return his/np.sum(his)
    
    # compute mean-std for training data
    def computeComponentPrior(self):
        dist = []
        for i in range(self.nClass):
            mean = np.mean(self.data[self.labels==i],axis=0)
            var = np.var(self.data[self.labels==i],axis=0)
            cov = np.cov(self.data[self.labels==i].T)
            cov = cov + np.diag(np.ones(len(mean))*self.smoothing)
            icov = np.linalg.pinv(cov)
            detcov = np.linalg.slogdet(cov)[1]
            
            dist.append([mean,cov,icov,detcov])     
        return dist

    def computeLogLikelihood_UnivariateGauss(self,xnew):
        joint_log_likelihood = []
        for i in range(self.nClass):
            mean = np.array(self.componentPrior[i][0])
            cov = np.array(self.componentPrior[i][1])
            invcov = np.array(self.componentPrior[i][2])
            detcov = np.array(self.componentPrior[i][3])
            
            jointy = np.log(self.classPrior[i])
            jointx = - 0.5 * ( np.log(2. * np.pi) * self.nDim + detcov )
            difmean = (xnew - mean)
            jointx -= 0.5 * np.dot(np.dot(difmean,invcov),difmean[np.newaxis].T)
            joint_log_likelihood.append(jointy + jointx)

        return joint_log_likelihood
          
    # predict class and compute prob of Xnew on each class
    def predict(self,xnew):
        probList = []
        #remove useless dimension if possible
        if (len(self.del_ind)!=0):
                xnew= np.delete(xnew,self.del_ind,axis=0)     
        joint_log_likelihood = self.computeLogLikelihood_UnivariateGauss(xnew)
        return [np.argmax(joint_log_likelihood),joint_log_likelihood]
    
    def testing(self,testing_data,testing_label):
        start = time.time()
        preLabels = []
        testing_data = DataPreprocessing.flattenImage(testing_data)
        for i in range(len(testing_data)):
            predictLabel, predictprob = self.predict(testing_data[i])
            preLabels.append(predictLabel)
        print('Mode: Single - Accuracy = {:f}% in {:f} seconds'.format(
                EvaluationMetrics.accuracy(preLabels,testing_label)*100,time.time()-start))
        
    def testingOptimized(self,testing_data,testing_label,bTestLLH=False,bLabel=False):
        start = time.time()
        testing_data = np.delete(DataPreprocessing.flattenImage(testing_data),self.del_ind,axis=1)
        joint_log_likelihood = []
        for i in range(self.nClass):
            mean = np.array(self.componentPrior[i][0])
            cov = np.array(self.componentPrior[i][1])
            invcov = np.array(self.componentPrior[i][2])
            detcov = np.array(self.componentPrior[i][3])
            
            jointy = np.log(self.classPrior[i])
            jointx = - 0.5 * ( np.log(2. * np.pi) * self.nDim + detcov)
            difmean = (testing_data - mean)
            jointx -= 0.5 * np.sum(np.dot(difmean,invcov)*difmean,axis=1)
            joint_log_likelihood.append(jointy + jointx)
        joint_log_likelihood = np.array(joint_log_likelihood)
        
        predLabels = np.argmax(joint_log_likelihood,axis=0)
        metricsVal = EvaluationMetrics.testingEvaluation([predLabels,joint_log_likelihood],testing_label,
                                                         self.nClass,start,'Batch',bTestLLH)
        if (bLabel==True):
            return predLabels
        return metricsVal