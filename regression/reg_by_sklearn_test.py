#coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


def test():
    #train_df = pd.read_csv("./input/train.csv", index_col=0)#use index 0
    train_df = pd.read_csv("./input/train.csv", index_col=0)
    test_df = pd.read_csv("./input/test.csv", index_col=0)
    train_df.head() #show top 5 lines

    #process SalePrice
    #matplotlib inline
    prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price+1)": np.log1p(train_df["SalePrice"])})
    prices.hist()
    ####prices.plot()
    #plt.show() #show photogram about prices

    y_train = np.log1p(train_df.pop('SalePrice'))
    """
    pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
       keys=None, levels=None, names=None, verify_integrity=False)
    objs: series，dataframe或者是panel构成的序列lsit
    axis： 需要合并链接的轴，0是行，1是列
    join：连接的方式 inner，或者outer
    """
    all_df = pd.concat((train_df, test_df), axis=0)
    #all_df.shape #==(2919,79)
    all_df['MSSubClass'].dtypes #(int64)
    all_df['MSSubClass'] = all_df['MSSubClass'].astype(str) #转换为str格式便于统计
    all_df['MSSubClass'].value_counts() #结果与.value_counts格式和内容不同, 类似于聚合

    #利用one-hot处理category类型列
    pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()
    #把所有的category数据，全部做one-hot处理
    all_dummy_df = pd.get_dummies(all_df)
    #all_dummy_df.head() #查看前5行

    #检测null值，求和并降序列出前10个
    #all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)

    mean_cols = all_dummy_df.mean() #求每列的均值
    all_dummy_df = all_dummy_df.fillna(mean_cols)
    #all_dummy_df.isnull().sum().sum() #check null unit

    #把数据集充新分为训练/测试集, df.doc() 可以选取需要的行/列
    dummy_train_df = all_dummy_df.loc[train_df.index] #获取全部train行
    dummy_test_df = all_dummy_df.loc[test_df.index] #获取全部test行

    #导入Ridge
    #from sklearn.linear_model import Ridge
    #from sklearn.model_selection import cross_val_score

    #transfer DF to Nmupy Array
    X_train = dummy_train_df.values #no ()
    X_test = dummy_test_df.values

    #numpy.logspace(start, stop, num, endpoint, base, dtype)
    #return an ndarray
    #num: the number of values between the range. Default is 50
    #endpoint: If true, stop is the last value in the range
    #base: Base of log space, default is 10
    alphas = np.logspace(-3, 2, 50)
    test_scores = []
    for alpha in alphas:
        reg = Ridge(alpha=alpha)
        test_score = np.sqrt(-cross_val_score(reg, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))

    plt.plot(alphas, test_scores)
    plt.title("Alpha vs CV Error")
    #plt.show()
    #看图得到alpha=15 误差最低
    #call Ridge get ridge result
    ridge = Ridge(alpha=15)
    ridge.fit(X_train, y_train)
    #log(1+x)->expm1()
    y_ridge = np.expm1(ridge.predict(X_test))

    #call random forest
    rf = RandomForestRegressor(n_estimators=500, max_features=0.3) #0.3*features总数
    rf.fit(X_train, y_train)
    y_rf = np.expm1(rf.predict(X_test))

    y_final = (y_ridge + y_rf) / 2
    #submission
    submission_df = pd.DataFrame(data={'Id':test_df.index, 'SalePrice':y_final})
    #import pdb; pdb.set_trace()
    #submission_df.head(10)

    print 1

if __name__ == "__main__":
    test()
