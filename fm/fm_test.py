#coding:UTF-8
 
from __future__ import division
from math import exp
from numpy import *
from random import normalvariate#正态分布
from datetime import datetime
 
trainData = '../input/diabetes_train.txt'
testData = '../input/diabetes_test.txt'
featureNum = 8
 
def loadDataSet(data):
    dataMat = []
    labelMat = []
    
    fr = open(data)#打开文件
    
    for line in fr.readlines():
        currLine = line.strip().split()
        #lineArr = [1.0]
        lineArr = []
        
        for i in xrange(featureNum):
            lineArr.append(float(currLine[i + 1]))
        dataMat.append(lineArr)
        
        labelMat.append(float(currLine[0]) * 2 - 1)
    return dataMat, labelMat
 
def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))
 
def stocGradAscent(dataMatrix, classLabels, k, iter):
    
    m, n = shape(dataMatrix)
    alpha = 0.01
    #initing
    w = zeros((n, 1))#n is num of labels
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    
    for it in xrange(iter):
        print it
        for x in xrange(m):#随机优化，对每一个样本而言的
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)#multiply对应元素相乘
            #完成交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            
            p = w_0 + dataMatrix[x] * w + interaction#计算预测的输出
        
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            print loss
        
            w_0 = w_0 - alpha * loss * classLabels[x]
            
            for i in xrange(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in xrange(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        
    
    return w_0, w, v
 
def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in xrange(m):
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)#multiply对应元素相乘
        #完成交叉项
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction#计算预测的输出
        
        pre = sigmoid(p[0, 0])
        
        result.append(pre)
        
        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue
        
    
    print result
    
    return float(error) / allItem
        
   
if __name__ == '__main__':
    dataTrain, labelTrain = loadDataSet(trainData)
    dataTest, labelTest = loadDataSet(testData)
    date_startTrain = datetime.now()
    print "Start train..."
    w_0, w, v = stocGradAscent(mat(dataTrain), labelTrain, 20, 200)
    print "Accuracy：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v))
    date_endTrain = datetime.now()
    print "Test during time：%s" % (date_endTrain - date_startTrain)
    print "start test:"
    print "Test Accuracy：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v))  
