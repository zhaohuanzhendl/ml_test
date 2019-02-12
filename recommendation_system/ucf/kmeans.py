#coding=utf8

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn import metrics 
import numpy as np

from datetime import datetime

#u1.base ref: https://grouplens.org/datasets/movielens/
u1_base = open("/Users/zhz/Downloads/ml-100k/u1.base")
#user id | item id | rating | timestamp. 

X_array = []
y_array = []

for l in u1_base:
    ls = l.strip().split("\t")
    #print ls
    y_array.append(int(ls[2]))
    dt = datetime.utcfromtimestamp(int(ls[3])).strftime("%Y-%m-%d")
    month = int(dt.split("-")[1])
    day = int(dt.split("-")[2])

    X_array.append([int(ls[0]), int(ls[1]), month, day])

#kmeans = KMeans(n_clusters=5, random_state=0).fit(X_array, y_array)
kmeans = MiniBatchKMeans(n_clusters=5, random_state=0).fit(X_array, y_array)

#print kmeans.labels_
#rint kmeans.predict(([[2,2,123345]]))
#print X_array[0:10]
#print y_array[:10]


#u1.testref: https://grouplens.org/datasets/movielens/
u1_test= open("/Users/zhz/Downloads/ml-100k/u1.test")

X_test = []
y_test = []

for l in u1_test:
    ls = l.strip().split("\t")
    y_test.append(int(ls[2]))
    dt = datetime.utcfromtimestamp(int(ls[3])).strftime("%Y-%m-%d")
    month = int(dt.split("-")[1])
    X_test.append([int(ls[0]), int(ls[1]), month, day])

y_pred = kmeans.predict(X_test)

for y in y_pred:
    y += 1 
print metrics.adjusted_rand_score(y_test, y_pred)
