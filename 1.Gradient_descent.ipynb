import numpy as np
import matplotlib.pyplot as plt

# 目標函數: y = x^2
def func(x): return np.square(x)

# 目標函數一階導數: dy/dx = 2*x
def dfunc(x): return 2 * x

def GD(x_start, df, epochs, lr):
    """ 梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param x_start: x的起始點
        :param df: 目標函數的一階導函數
        :param epochs: 反覆運算週期
        :param lr: 學習率
        :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1
     """
    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)              #計算當前位置的梯度
        v = - dx * lr           #計算更新步伐
        x += v                  #更新 x 的值
        xs[i+1] = x             #記錄更新後的位置
    return xs

# --- 主程式設定 ---
x_start = 5  # 起始權重
epochs = 20  # 執行週期數(15->20)
lr = 0.3     # 學習率

# 執行梯度下降法
x = GD(x_start, dfunc, epochs, lr=lr)
print(x)

color = 'r'
from numpy import arange
t = arange(-6.0, 6.0, 0.01)

# 繪製二次函數曲線與梯度下降軌跡
plt.plot(t, func(t), c='b', label='y=x^2')                 # 藍色線條代表目標函數
plt.plot(x, func(x), c=color, label='lr={}'.format(lr))   # 紅色線條連接更新軌跡
plt.scatter(x, func(x), c=color)                         # 標示出每個點
plt.legend()
plt.show()
