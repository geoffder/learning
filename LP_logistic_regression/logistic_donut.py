import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

R_inner = 5
R_outer = 10

R1 = np.random.randn(N//2) + R_inner # // is integer division in py3
theta = 2 * np.pi * np.random.random(N//2) # polar coordinates
# use polar coordinates to create (x,y) coordinates. X_inner is 2 columns, x and y.
X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2 * np.pi * np.random.random(N//2) # polar coordinates, again, new random values
X_outer = np.concatenate([[R2*np.cos(theta)], [R2*np.sin(theta)]]).T

# X_inner with the smaller radius, will be surrounded by X_outer
# combine in to one set
X = np.concatenate((X_outer, X_inner), axis=0)
T = np.array([0]*(N//2) + [1]*(N//2)) # inner is class 1, outer is class 0

# take a look at the donut dataset
# plt.scatter(X[:,0], X[:,1], c=T, alpha=.5)
# plt.show()

ones = np.ones((N,1)) # for bias column
# to solve the donut problem, we will create yet another column
# this extra column will represent the radius of each point
r = np.array([np.sqrt(X[i,:] @ X[i,:]) for i in range(N)]).reshape(N,1)
# if building r this way, need to reshape from (N,) to (N,1)
Xb = np.concatenate((ones, r, X), axis=1)

w = np.random.randn(D+2) / np.sqrt(D)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def crossEntropy(Y, T):
    return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).mean()

def costDeriv(X, Y, T):
    return X.T @ (Y - T)

def prediction_rate(P, T):
    return (P == T).mean()

learning_rate = .001
l2 = 1 # L2 regularization constant
costs = []
for i in range(5000):
    Y = sigmoid(Xb @ w)
    costs.append(crossEntropy(Y, T))
    w -= learning_rate * (costDeriv(Xb, Y, T) + l2 * w)

Y = sigmoid(Xb @ w)
costs.append(crossEntropy(Y, T))
P = np.round(Y) # final predictions

print('w:', w)
print('prediction rate:', prediction_rate(P, T))

plt.plot(costs)
plt.show()

plt.scatter(X[:,0], X[:, 1], c = P)
plt.show()
