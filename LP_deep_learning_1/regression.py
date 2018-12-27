import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 600
D = 2
M = 5

X = np.random.randn(N, D)
T = X[:,0] * X[:,1]

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(X[:,0], X[:,1],T)
# plt.show()

def forward(X, W1, b1, W2, b2):
    z = X @ W1 + b1
    Z = np.tanh(z) #tanh
    A = Z @ W2 + b2
    return A, Z

def cost(T, Y):
    return (T - Y).T @ (T - Y)

def deriv_w2(Z, T, Y):
    return Z.T @ (T - Y)

def deriv_b2(T, Y):
    return (T - Y).sum() #linear regression, one output

def deriv_w1(X, Z, T, Y, W2):
    return X.T @ (np.outer(T - Y, W2) * (1 - Z*Z))

def deriv_b1(T, Y, W2, Z):
    return (np.outer(T - Y, W2) * (1 - Z*Z)).sum() #linear regression, one output

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M)
b2 = np.random.randn(1)

learning_rate = 10e-6
costs = []
for epoch in range(50000):
    Y, Z = forward(X, W1, b1, W2, b2)
    if epoch % 100 == 0:
        c = cost(T, Y)
        print('cost:', c)
        costs.append(c)

    W2 += learning_rate * deriv_w2(Z, T, Y)
    b2 += learning_rate * deriv_b2(T, Y)
    W1 += learning_rate * deriv_w1(X, Z, T, Y, W2)
    b1 += learning_rate * deriv_b1(T, Y, W2, Z)

plt.plot(costs)
plt.show()

#plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0], X[:,1],T)

# create a meshgrid and predict Y values using trained network
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Y, _ = forward(Xgrid, W1, b1, W2, b2)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Y, linewidth=.2, antialiased=True)
plt.show()

# plot magnitude of residuals (it does the worst on the corners and pockets around the middle)
Ygrid = Xgrid[:,0] * Xgrid[:,1]
R = np.abs(Ygrid - Y)
plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
plt.show()

# plot the residuals as a meshgrid
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=.2, antialiased=True)
plt.show()
