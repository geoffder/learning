import numpy as np
import matplotlib.pyplot as plt

Nclass = 500 # 500 samples per class

# make 3 gaussian clouds
X1 = np.random.randn(Nclass, 2) + np.array([0, -2]) # centred at (0, -2)
X2 = np.random.randn(Nclass, 2) + np.array([2, 2]) # centred at (2, 2)
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2]) # centred at (-2, 2)

X = np.concatenate((X1,X2,X3), axis=0)
T = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

#plt.scatter(X[:,0], X[:,1], c=Y, alpha=.5)
#plt.show()

D = 2 # two dimentions / features
M = 3 # three nodes in the hidden layer
K = 3 # three output classes

W1 = np.random.randn(D, M) # from inputs to hidden layer
b1 = np.random.randn(M)
W2 = np.random.randn(M, K) # from hidden layer to outputs
b2 = np.random.randn(K)

def softmax(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X @ W1 - b1)) # sigmoid #should be negative? check github
    A = Z @ W2 + b2
    # A = np.tanh(X @ W1 + b1) @ W2 + b2 # I did this, LP wants to use sigmoid
    return (softmax(A))

def classification_rate(T, Y):
    return (np.argmax(Y, axis=1) == T).mean()

Y = forward(X, W1, b1, W2, b2)
print('prediction dimentions:', Y.shape) # NxK matrix
print(Y.sum(axis=1)) # each row (sample) of predictions sums to 1

print('classification rate:,', classification_rate(T, Y)) # ~33% because it is untrained.
