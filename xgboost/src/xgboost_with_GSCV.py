#coding=utf-8
#homesite competition address:
#https://www.kaggle.com/c/homesite-quote-conversion/data
#download train.csv/test.csv from the ref-address

import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import *

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head()

train = train.drop('QuoteNumber', axis=1)
test = test.drop('QuoteNumber', axis=1)

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop('Original_Quote_Date', axis=1)

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

train['Year'] = train['Date'].apply(lambda x: x.year)
train['Month'] = train['Date'].apply(lambda x: x.month)
train['weekday'] = train['Date'].apply(lambda x: x.weekday())

test['Year'] = test['Date'].apply(lambda x: x.year)
test['Month'] = test['Date'].apply(lambda x: x.month)
test['weekday'] = test['Date'].apply(lambda x: x.weekday())

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

#train.ix[:, train.isnull().any()]
#fill -999 to NAs
train = train.fillna(-999)
test = test.fillna(-999)

#check train/test columns diff
#a = set(train.columns)
#b = set(test.columns)
#a.difference(b)

features = list(train.columns[1:])  #la colonne 0 est le quote_conversionflag
print(features)

for f in train.columns:
    if train[f].dtype=='object':
        print(f) #f columns name/ index
        lbl = preprocessing.LabelEncoder()
        #train[f].values : <type 'numpy.ndarray'>
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                   #cv=StratifiedKFold(train['QuoteConversion_Flag'], n_folds=5, shuffle=True),
                   cv=5,
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(train[features], train["QuoteConversion_Flag"])

#import pdb; pdb.set_trace()
#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(test[features])[:,1]

sample = pd.read_csv('../input/sample_submission.csv')
sample.QuoteConversion_Flag = test_probs
sample.to_csv("xgboost_best_parameter_submission.csv", index=False)
