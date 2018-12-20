import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 4
D = 2

# logical combinations
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

T = np.array([0, 1, 1, 0])#.reshape(N,1)

# look at the problem
# plt.scatter(X[:,0], X[:,1], c=T)
# plt.show() # can't cut any direction with a straight line to get good classification

# we'll solve the XOR problem by addiing a third coordinate dimension
ones = np.ones((N,1))
xy = (X[:,0] * X[:,1]).reshape(N, 1)
Xb = np.concatenate((ones, xy, X), axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def crossEntropy(Y,T):
    return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).mean()

def costDeriv(X, Y, T):
    return X.T @ (Y - T)

def prediction_rate(P, T):
    return (P == T).mean()

learning_rate = .01
l2 = .01 # L2 lambda
w = np.random.randn(D+2) / np.sqrt(D+2)

costs = []
for i in range(3000):
    Y = sigmoid(Xb @ w)
    costs.append(crossEntropy(Y, T))
    w -= learning_rate * (costDeriv(Xb, Y, T) + l2 * w)

Y = sigmoid(Xb @ w)
costs.append(crossEntropy(Y, T))
P = np.round(Y)

print('w:', w)
print('prediction rate:', prediction_rate(P, T))

plt.plot(costs)
plt.show()

plt.scatter(X[:,0], X[:,1], c=P)
plt.show()

# 3D for fun
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(Xb[:,1], Xb[:,2],Xb[:,3], c = P)
# plt.show()
