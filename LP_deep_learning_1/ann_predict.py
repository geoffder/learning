import numpy as np
import matplotlib.pyplot as plt

from process import get_data

X, T = get_data()
N, D = X.shape

M = 5 # hidden layer nodes
K = len(set(T)) # number of unique values in T #int(T.max() + 1)

W1 = np.random.randn(D, M)
b1 = np.zeros(M) #np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K) #np.random.randn(K)

def softmax(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    #Z = 1 / (1 + np.exp(X @ W1 + b1)) # using tanh this time
    Z = np.tanh(X @ W1 + b1)
    A = Z @ W2 + b2
    return softmax(A)

def classification_rate(T, P):
    return (T == P).mean()

Y = forward(X, W1, b1, W2, b2)
P = np.argmax(Y, axis=1)

print('classification rate:', classification_rate(T, P))
