import numpy as np
import matplotlib.pyplot as plt

# fat (T H I C C) matrix
N = 50
D = 50

# uniform dist from -5 to 5 centred on zero
X = (np.random.random((N, D)) - .5)*10
# first three terms the only ones that influence output
true_w = np.array([1, .5, -.5] + [0]*(D-3))
# Y shaped by the first three columns of X plus gaussian noise
Y = X.dot(true_w) + np.random.randn(N) * .5

costs_nonreg = []
costs_l1 = []
w_nonreg = np.random.randn(D) / np.sqrt(D) # mu = 0, variance = 1/D (stdev = sqrt(variance))
w_l1 = w_nonreg[:] # start with same values
learning_rate = .001
l1 = 10.0
tries = 500

def costDeriv(w):
    return X.T @ (X @ w - Y)

for i in range(tries):
    w_nonreg = w_nonreg - learning_rate * costDeriv(w_nonreg)
    w_l1 = w_l1 - learning_rate * (costDeriv(w_l1) + l1 * np.sign(w_l1))

    Yhat_nonreg = X @ w_nonreg
    Yhat_l1 = X @ w_l1

    costs_nonreg.append(np.dot(Yhat_nonreg - Y, Yhat_nonreg - Y) / N)
    costs_l1.append(np.dot(Yhat_l1 - Y, Yhat_l1 - Y) / N)

# how does the data fit look
plt.plot(Y, label = 'target')
plt.plot(Yhat_l1, label = 'l1')
plt.plot(Yhat_nonreg, label = 'non-reg')
plt.legend()
plt.show()
# compare calculated w to the true w
plt.plot(true_w, label = 'true w')
plt.plot(w_l1, label = 'l1')
plt.plot(w_nonreg, label = 'non-reg')
plt.legend()
plt.show()
# compare costs between non regularized and l1 regularized method
plt.plot(costs_nonreg, label = 'non regularized')
plt.plot(costs_l1, label = 'l1 regularized')
plt.legend()
plt.show()
