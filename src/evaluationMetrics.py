import numpy as np
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from collections import Counter
from sklearn.metrics import confusion_matrix

class EvaluationMetrics(object):
    @staticmethod
    def roundByClass(y,classID):
        y[y<classID[0]] = classID[0]
        y[y>classID[-1]] = classID[-1]
        return y
    
    @staticmethod
    def scatterPlotRegression(yLabels,yPredict,numberOfPoints=2000):
        # get small part of training to plot 
        idx = np.random.choice(len(yLabels),numberOfPoints)
        classIDs = np.array(list(set(yLabels)))
        colors = cm.rainbow(np.linspace(0, 1, len(classIDs)))
        #get data
        yLabels = yLabels[idx]
        yPredict = EvaluationMetrics.roundByClass(yPredict[idx],classIDs)
        yPredictRound = EvaluationMetrics.roundByClass(np.around(yPredict),classIDs)

        maxheight = np.max(yPredict)
        #statistic about prediction
        from IPython.display import display
        display(pd.DataFrame(list(Counter(yPredictRound).items()),columns=['ClassID','Quatity on prediction']))

        plt.figure(figsize=(20,10))
        plt.ylabel('Y prediction')
        plt.xlabel('Y real')
        plt.ylim(-0.5,maxheight+0.5)
        plt.xlim(-0.5,classIDs[-1]+0.5)
        plt.xticks(classIDs)
        plt.yticks(classIDs)
        #plt.ylim(plotMean-3*plotStd,plotMean+3*plotStd)
        for idx,classid in enumerate(classIDs):
            plt.bar(idx-0.5,maxheight+0.5,width=1,color=colors[idx],alpha=0.5,zorder=1)
            x = yLabels[yPredictRound == classid]
            y = yPredict[yPredictRound == classid]
            plt.scatter(x,y,c=colors[idx],lw=0.1,s=100,label=classid,zorder=2,edgecolor='white')
        plt.legend(fontsize=15,loc=0,bbox_to_anchor=(1, 0.5))
        
    @staticmethod
    def accuracy(labels,real_labels):
        return np.sum(np.array(labels)==real_labels)/len(labels)
    
    @staticmethod
    def approximateLogSum(x):
        maxValue = np.max(x,axis=0)
        return maxValue + np.log(np.sum(np.exp(x-maxValue),axis=0))
    
    @staticmethod
    def testlikelihood(llh,labels,nClass):
        nTest = len(labels)
        labels = np.array(labels).astype(int)
        labelMatrix = np.zeros([nClass,nTest])
        
        for i in range(nTest):
            labelMatrix[labels[i],i]=1.
        
        #print(EvaluationMetrics.approximateLogSum(llh)[0])
        return np.sum((llh - EvaluationMetrics.approximateLogSum(llh))*labelMatrix)/nTest 
       
    @staticmethod
    def testingEvaluation(preLabels,testing_label,nClass,start,mode,bTestLLH=True):
        fAccuracy = EvaluationMetrics.accuracy(preLabels[0],testing_label)*100
        fTestllh = []
        # print
        print('Mode: {} - Accuracy = {:f}%'.format(mode,fAccuracy),end=' ')
        if (bTestLLH):
            fTestllh = EvaluationMetrics.testlikelihood(preLabels[1],testing_label,nClass)
            print('- Test likelihood = {:f}'.format(fTestllh),end=' ')
        print('in {:f} seconds'.format(time.time()-start))
        
        np.set_printoptions(threshold=np.nan)  
        #print(preLabels[0][0:20])#,preLabels[1][:,0])
        return [fAccuracy,fTestllh] 
    
    
    
    @staticmethod
    def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues,fontsize=15):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        printStr='{:.0f}'
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            printStr = '{:.3f}'
            
        plt.rcParams.update({'font.size': fontsize})
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title,fontsize=fontsize)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, printStr.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.grid(False)

    @staticmethod
    def plotConfusionMatrixModel(y_predict,y_true):
        cnf_matrix = confusion_matrix(y_predict, y_true)
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        EvaluationMetrics.plot_confusion_matrix(cnf_matrix, classes=np.arange(10),
                              title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.subplot(1,2,2)
        EvaluationMetrics.plot_confusion_matrix(cnf_matrix, classes=np.arange(10), normalize=True,
                              title='Normalized confusion matrix')
        plt.show()
    
    @staticmethod
    def MSE(yPre,yReal):
        return ((yPre - yReal) ** 2).mean()