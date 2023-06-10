 # Thêm thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time





# Load data từ file csv
data = pd.read_csv('data_class_1.csv').values
N, d = data.shape
x = data[:, 0:d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)

# Vẽ data bằng scatter
plt.scatter(x[:10, 0], x[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x[10:, 0], x[10:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend(loc=1)
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')
t_run = time.time()
# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Thêm cột 1 vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x))
# print (x)

w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
# print("weight")
# print(w)
# print (w)
# Số lần lặp bước 2
numOfIteration = 300
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.01

for i in range(1, numOfIteration):
    # Tính giá trị dự đoán
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
    plt.plot(cost)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
    # Gradient descent
    grad = np.dot(x.T, y_predict - y)
    # accuracy_score(y, np.round(abs(y_predict)), normalize=False)
    w = w - learning_rate * np.dot(x.T, y_predict - y)
# print(w)
# Vẽ đường phân cách.
# t = 0.5
t_off = time.time()
time_run = t_off - t_run
print(time_run)
# plt.plot((4, 10), (-(w[0] + 4 * w[1] + np.log(1 / t - 1)) / w[2], -(w[0] + 10 * w[1] + np.log(1 / t - 1)) / w[2]), 'g')
values = np.arange(36, 52, 0.1)

z = values + np.full(values.shape, w[0])
print(z)
# print(cost)
# plt.plot(cost)
plt.plot(z, sigmoid(z), 'g')

plt.show()