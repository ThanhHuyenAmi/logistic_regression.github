import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time
data = pd.read_csv('data_class_1.csv', header = None)
true_x = []
true_y = []
false_x = []
false_y = []
# print(data.values)
for item in data.values:
    if item[2] == 1.:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])
plt.scatter(true_x, true_y, marker='o', c='b')
plt.scatter(false_x, false_y, marker='s', c='r')


def sigmoid(z):
    return 1/(1+np.exp(-z))
"Hàm phân chia"
def divide(p):
    if p >= 0.5:
        return 1
    else:
        return 0

def predict(feature, weights):
    z = np.dot(feature, weights)
    """
    VD: bias = 3 + 4x6 + 6x8
    hàm dot trong numpy: 3x1 + 4x6 + 6x8
    -> [3,4,6] dot [1,6,8]
    """
    return sigmoid(z)
"Hàm chi phí để tối ưu"
def cost_funtion(feature, labels, weights):
    """
    :param feature:  mảng (100 x 3)
    :param labels: (100x1) giá trị 1 hoặc 0
    :param weights: (3x1)
    prediction: chứa giá trị dự đoán
    tuy nhiên prediction vẫn chứa giá trị lẫn lộn nhãn bằng 0
    ma trận chuyển vị: chuyển từ ma trận hàng thành cột
    :return: chi phí
    """
    # n = len(labels)
    predictions = predict(feature, weights)
    cost_class1 = np.multiply(labels, np.log(predictions))
    cost_class2 = np.multiply((1-labels), np.log(1-predictions))
    cost = - (cost_class1 + cost_class2)
    return cost.sum()
def update_weight(feature, labels , weights,  learning_rate):
    """
    :param feature: (100x3)
    :param weights: (100x1)
    :param learning_rate: float
    :return: new weights float
    """
    # n = len(labels)
    predictions = predict(feature, weights)
    grad = np.dot(feature.T, (predictions - labels))
    grad = grad * learning_rate
    weights = weights - grad
    # print(weights)
    return weights
def train(feature, labels, weights, learning_rate, iter):
    cost_hs = []
    # cost_hs = np.zeros((iter, 1))
    for i in range(iter):
        weights = update_weight(feature, labels, weights,  learning_rate)
        cost = cost_funtion(feature, labels, weights)
        cost_hs.append(cost)
        plt.plot(cost_hs)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()


    return weights, cost_hs
label = data.values[: , 2:3]
x = data.values[:, :2]
arr = np.ones([20,1])
# arr_2 = np.zeros([20,1])
x_1 = insert(x, [0], arr, 1)
# print(x_1.shape)
t_start = time.time()

weight = ([0.], [0.1], [0.1])
weight, cost_hs_1 = train(x_1, label, weight, 0.01, 300)
# print('ket qua:')
# print(weight)
# print(cost_hs_1)
# print("gia tri du doan")
# print(predict([1, 7, 0.15], weight))
# print(divide(predict([1, 7, 0.15], weight)))
# plt.plot((4, 10), (-(weight[0] + 4 * weight[1] + np.log(1 / t - 1)) / weight[2], -(weight[0] + 10 * weight[1] + np.log(1 / t - 1)) / weight[2]), 'g')
values = np.arange(-5, 20, 0.1)
z = values + np.full(values.shape, weight[0])
t_off = time.time()
t_run = t_off - t_start
print(t_run)
# X = range(1, len(cost_hs_1) + 1)
# plt.plot(X, cost_hs_1)
plt.plot(z, sigmoid(z), 'g')
plt.show()

