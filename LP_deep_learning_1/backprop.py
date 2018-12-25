import numpy as np
import matplotlib.pyplot as plt

def softmax(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X @ W1 - b1)) # sigmoid
    A = Z @ W2 + b2
    return (softmax(A), Z)

def classification_rate(T, P):
    return (T == P).mean()

def cost(T, Y):
    return (T * np.log(Y)).sum()

# slow versions for easier visualization
def slow_deriv_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]

    ret1 = np.zeros((M, K))
    for n in range(N):
        for m in range(M):
            for k in range(K):
                ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]
    return ret1

def slow_deriv_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape

    ret1 = np.zeros((D, M))
    for n in range(N):
        for k in range(K):
            for m in range(M):
                for d in range(D):
                    ret1[d,m] += (T[n,k] - Y[n,k])*W2[m,k] *Z[n,m] *(1 - Z[n,m]) *X[n,d]
    return ret1

def deriv_w2(Z, T, Y):
    # (NxM).T @ (NxK) = (MxK)
    return Z.T @ (T - Y)

def deriv_b2(T, Y):
    # (NxK).sum(axis=0) = (Kx1)
    return (T - Y).sum(axis=0) # one per class

def deriv_w1(X, Z, T, Y, W2):
    # (NxD).T @ ((NxK) @ (MxK).T * (NxM) * (1 - NxM))
    # (NxD).T @ ((NxM) * (NxM) * (1 - NxM))
    # (DxM), correct dimentions for W1 (input weights)
    return X.T @ ((T - Y) @ W2.T * Z * (1 - Z))

def deriv_b1(T, Y, W2, Z):
    # ((NxK) @ ((MxK).T * (NxM) * (1 - NxM))).sum(axis=0)
    # ((NxM) * (NxM) * (1 - NxM)).sum(axis=0) = (Mx1)
    return ((T - Y) @ W2.T * Z * (1 - Z)).sum(axis=0)

def main():
    # create the data
    Nclass = 500
    D = 2 # number of input features / dimensionality
    M = 3 # hidden layer size
    K = 3 # number of classes

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)

    T = np.zeros((N, K))
    T[np.arange(N), Y.astype(np.int32)] = 1

    #plt.scatter(X[:,0], X[:,1], c=Y, alpha=.5)
    #plt.show()

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-7
    costs = []
    for epoch in range(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print('cost:', c, 'classification rate:', r)
            costs.append(c)

        W2 += learning_rate * deriv_w2(hidden, T, output)
        b2 += learning_rate * deriv_b2(T, output)
        W1 += learning_rate * deriv_w1(X, hidden, T, output, W2)
        b1 += learning_rate * deriv_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()

    plt.scatter(X[:,0], X[:,1], c=P, alpha=.5)
    plt.show()
if __name__ == '__main__':
    main()
