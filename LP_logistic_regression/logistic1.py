import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# here we'll calculate the sigma sigmoid

N = 100
D = 2

X = np.random.randn(N,D)
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis = 1) # column vector of ones for bias term

w = np.random.randn(D+1)

z = Xb @ w
print(Xb@w == Xb.dot(w))
def sigmoid(z):
 return 1. / (1. + np.exp(-z))

sns.lineplot(x = range(N), y = sigmoid(z))
plt.show()
