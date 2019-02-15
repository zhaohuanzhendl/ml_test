#coding=utf-8

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 = []
    for t in dataSet:
        for item in t:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #import pdb;pdb.set_trace()
    return map(frozenset, C1)


def scanD(D, Ck,  minSupport):
    '''根据候选集ck， 最小支持度minSupport 生成频繁集Lk
    '''
    ssCnt = dict()
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can] = 1 
                else: ssCnt[can] += 1

    numItems = float(len(D))
    retList = list()
    supportData = dict()
    for key, cnt in ssCnt.items():
        support = cnt/numItems
        #if cnt/numItems >= minSupport:
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    
    return retList, supportData


def aprioriGen(Lk, k):
    '''创建候选集Ck
    '''
    retList = list()
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet,  minSupport=0.5):
    #import pdb; pdb.set_trace()
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2

    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1

    return L, supportData


#createC1(loadDataSet())
apriori(loadDataSet())
