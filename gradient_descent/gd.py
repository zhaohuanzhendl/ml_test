#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
beta = [1, 1]
alpha = 0.2
tol_L = 0.1
batch_size = 16

max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

def compute_grad(beta, x, y):
    grad = [0, 0]
    grad[0] = 2.0 * np.mean(beta[0] + beta[1]*x -y)
    grad[1] = 2.0 * np.mean(x*(beta[0] + beta[1]*x -y))
    return np.array(grad)


def compute_grad_batch(beta, batch_size, x, y):
    #import pdb
    #pdb.set_trace()
    grad = [0, 0]
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2.0 * np.mean(beta[0] + beta[1]*x[r] -y[r])
    grad[1] = 2.0 * np.mean(x[r]*(beta[0] + beta[1]*x[r] -y[r]))
    return np.array(grad)

def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta


def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1]*x -y) **2
    res = np.sqrt(np.mean(squared_err))
    return res

np.random.seed(100)
grad = compute_grad(beta, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)

"""
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad(beta, x, y)
    loss = loss_new
    loss_new = rmse(beta, x, y)
    i += 1
    print ('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))

print ('Coef: %s\nIntercept %s'%(beta[1], beta[0]))
"""

i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    #grad = compute_grad(beta, x, y)
    grad = compute_grad_batch(beta, batch_size, x, y)
    if i % 100 == 0:
        loss = loss_new
        loss_new = rmse(beta, x, y)
        print ('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
    i += 1

print ('Coef: %s\nIntercept %s'%(beta[1], beta[0]))





