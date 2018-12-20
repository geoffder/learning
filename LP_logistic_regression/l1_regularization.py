import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

N = 50
D = 50 # extra T H I C C matrix

# np.random.random creates uniform dist from 0 to 1
X = (np.random.random((N, D)) - .5) * 10 # -.5 to centre on zero. Range now -5 to 5
ones = np.ones((N,1))
Xb = np.concatenate((X, ones), axis=1) # add bias column
true_w = np.array([1, .5, -.5] + [0]*(D-2)) # only first three dimensions actually matter, add bias
T = np.round(sigmoid(Xb @ true_w + np.random.randn(N)*.5)) # targets, shaped by true w, plus noise

learning_rate = .0001
l1 = 4 # L1 regularization lambda
w = np.random.randn(D+1) / np.sqrt(D+1) # initialize random weights

def costDeriv(X, T, Y):
    return X.T @ (Y - T)

def crossEntropy(Y, T, w):
    return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).mean() + l1*np.abs(w).mean()

costs = []
for i in range(1000):
    Y = sigmoid(Xb @ w)
    costs.append(crossEntropy(Y, T, w))
    w -= learning_rate * (costDeriv(Xb, T, Y) + l1 * np.sign(w))


plt.plot(costs)
plt.show()

plt.plot(true_w)
plt.plot(w)
plt.show()
