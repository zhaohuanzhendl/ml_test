# -*- coding=utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 产生数据
def get_data(num_data=100):
    x = np.reshape(np.random.normal(1, 1, num_data), (num_data, 1))
    y = np.reshape(np.random.normal(0, 1, num_data), (num_data, 1))
    bias = np.ones((num_data, 1))
    class1 = np.concatenate((x, y, bias), axis=1)
    x = np.reshape(np.random.normal(5, 1, num_data), (num_data, 1))
    y = np.reshape(np.random.normal(6, 1, num_data), (num_data, 1))
    class2 = np.concatenate((x, y, bias), axis=1)

    plt.plot(class1[:, 0], class1[:, 1], 'rs', class2[:, 0], class2[:, 1], 'go')
    plt.grid(True)
    plt.title('Distribution')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    label_data = np.zeros((2*num_data, 1))
    label_data[num_data:2*num_data] = 1.0

    return np.concatenate((class1[:, :], class2[:, :]), axis=0), label_data


# sigmoid 函数
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))


# 梯度下降法
def grad_ascent(train_data, label_data, num_iter, alpha=0.001):
    weights = np.ones((train_data.shape[1], 1))
    train_data = np.mat(train_data)
    label_data = np.mat(label_data)
    weights_x1 = []
    weights_x2 = []
    weights_bias = []
    for i in np.arange(num_iter):
        temp = sigmoid(train_data*weights)
        error = label_data - temp
        weights = weights + alpha * train_data.T * error
        weights_x1.append(weights[0])
        weights_x2.append(weights[1])
        weights_bias.append(weights[2])

    weights = np.array(weights)
    weights = weights[:, 0]

    # 显示参数变化曲线
    x = np.arange(num_iter)
    weights_x1 = np.array(weights_x1)[:, 0, 0]
    weights_x2 = np.array(weights_x2)[:, 0, 0]
    weights_bias = np.array(weights_bias)[:, 0, 0]
    plt.subplot(311)
    plt.plot(x, weights_x1, 'b-')
    plt.title('weight_x1')
    plt.grid(True)
    plt.subplot(312)
    plt.plot(x, weights_x2, 'b-')
    plt.title('weight_x2')
    plt.grid(True)
    plt.subplot(313)
    plt.plot(x, weights_bias, 'b-')
    plt.title('weight_bias')
    plt.grid(True)
    plt.show()

    return weights


# 随机梯度下降法
def stoc_grad_ascent(train_data, label_data, num_iter, alpha=0.001):
    weights = np.ones((1, train_data.shape[1]))
    train_data = np.mat(train_data)
    label_data = np.mat(label_data)
    weights_x1 = []
    weights_x2 = []
    weights_bias = []

    for i in np.arange(num_iter):
        for j in np.arange(train_data.shape[0]):
            temp = sigmoid(weights*train_data[j, :].T)
            error = label_data[j][0] - temp[0]
            weights = weights + alpha*error*train_data[j, :]
            weights_x1.append(weights[0, 0])
            weights_x2.append(weights[0, 1])
            weights_bias.append(weights[0, 2])

    weights = np.array(weights)
    weights = weights[0, :]
    # 显示参数变化曲线
    x = np.arange(num_iter*train_data.shape[0])
    weights_x1 = np.array(weights_x1)
    weights_x2 = np.array(weights_x2)
    weights_bias = np.array(weights_bias)
    plt.subplot(311)
    plt.plot(x, weights_x1, 'b-')
    plt.title('weight_x1')
    plt.grid(True)
    plt.subplot(312)
    plt.plot(x, weights_x2, 'b-')
    plt.title('weight_x2')
    plt.grid(True)
    plt.subplot(313)
    plt.plot(x, weights_bias, 'b-')
    plt.title('weight_bias')
    plt.grid(True)
    plt.show()

    return weights


# 改进的随机梯度下降法
def stoc_grad_ascent1(train_data, label_data, num_iter):
    weights = np.ones((1, train_data.shape[1]))
    train_data = np.mat(train_data)
    label_data = np.mat(label_data)
    weights_x1 = []
    weights_x2 = []
    weights_bias = []
    data_index = np.arange(train_data.shape[0])

    for i in np.arange(num_iter):
        for j in np.arange(train_data.shape[0]/10):
            alpha = 0.0/(1.0+i+j) + 0.000009
            temp = sigmoid(weights*train_data[j*10:(j+1)*10, :].T)
            error = label_data[j][0] - temp[0]
            weights = weights + alpha*error*train_data[j*10:(j+1)*10, :]
            weights_x1.append(weights[0, 0])
            weights_x2.append(weights[0, 1])
            weights_bias.append(weights[0, 2])

    weights = np.array(weights)
    weights = weights[0, :]
    # 显示参数变化曲线
    x = np.arange(num_iter*train_data.shape[0]/10)
    weights_x1 = np.array(weights_x1)
    weights_x2 = np.array(weights_x2)
    weights_bias = np.array(weights_bias)
    plt.subplot(311)
    plt.plot(x, weights_x1, 'b-')
    plt.title('weight_x1')
    plt.grid(True)
    plt.subplot(312)
    plt.plot(x, weights_x2, 'b-')
    plt.title('weight_x2')
    plt.grid(True)
    plt.subplot(313)
    plt.plot(x, weights_bias, 'b-')
    plt.title('weight_bias')
    plt.grid(True)
    plt.show()

    return weights


# 画出决策直线
def plot_decision(train_data, data_num, weights):
    x = np.linspace(-2, 9, train_data.shape[0])
    y = np.array((-weights[2]-weights[0]*x)/weights[1])

    plt.plot(train_data[0:data_num, 0], train_data[0:data_num, 1], 'bs',
             train_data[data_num:2*data_num, 0], train_data[data_num:2*data_num, 1], 'go',
             x, y, 'r-')
    plt.grid(True)
    plt.title('line')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    data_num = 100
    train_data, train_label = get_data(data_num)
    # 梯度下降法
    weights = grad_ascent(train_data, train_label, 1000)

    # 随机梯度下降法
    # weights = stoc_grad_ascent(train_data, train_label, 400)

    # 改进的随机梯度下降法
    #weights = stoc_grad_ascent1(train_data, train_label, 800)
    # 画出决策直线
    plot_decision(train_data, data_num, weights)
