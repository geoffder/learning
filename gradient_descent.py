import numpy as np
import matplotlib.pyplot as plt

N = 10 # data points
D = 3 # dimensionality

# initialize X
X = np.zeros((N, D)) # (N, D) is one argument, that's why 2 brackets
X[:,0] = 1 # bias term
X[:5, 1] = 1 # first 5 ele of first column to 1
X[5:, 2] = 1 # last five elements of the third column to 1

# initialize Y
Y = np.array([0]*5 + [1]*5) # [0,0,0,0,0,1,1,1,1,1]

# if you try to do the usual w solution it fails
try:
    w = np.linalg.solve(X.T @ X, X.T @ Y)
except Exception as e:
    print("w calculation with numpy.linalg failure:", e)

# instead we will use gradient descent to calculate w
costs = []
w = np.random.randn(D) / np.sqrt(D) # mu = 0, variance = 1/D
print("initial weights:", w)
learning_rate = .001
tries = 1000

def costDeriv(w):
    return X.T @ (X @ w - Y)
    #return X.T @ (X @ w) - X.T @ Y

#print((X.T @ (X @ w - Y)) == (X.T @ (X @ w) - X.T @ Y))

for i in range(tries):
    w = w - learning_rate * costDeriv(w)
    Yhat = X @ w
    mse = (Y - Yhat).T @ (Y - Yhat)
    costs.append(mse)

print("final weights:", w)

plt.plot(costs)
plt.show()

plt.plot(Y, label='target')
plt.plot(Yhat, label='prediction')
plt.legend()
plt.show()
