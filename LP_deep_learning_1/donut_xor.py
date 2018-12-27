import numpy as np
import matplotlib.pyplot as plt

# binary classification no softmax!

def forward(X, W1, b1, W2, b2):
     # sigmoid
    # Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))

    # tanh
    Z = np.tanh(X @ W1 + b1)

    # relu
    # Z = X.dot(W1) + b1
    # Z = Z * (Z > 0)

    activation = Z @ W2 + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z

def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)

def derivative_w2(Z, T, Y):
    # Z is (NxM)
    return Z.T @ (T - Y)

def derivative_b2(T, Y):
    return (T - Y).sum() # only one output, scalar

def derivative_w1(X, Z, T, Y, W2):
    #w1 = X.T @ (np.outer(T - Y, W2) * Z *(1 - Z)) # sigmoid deriv
    #w1 = X.T @ ((T - Y) @ W2.T * Z * (1 - Z)) # sigmoid deriv
    #w1 = X.T @ ((T - Y) @ W2.T * (1 - Z * Z)) # tanh deriv
    w1 = X.T @ (np.outer(T - Y, W2) * (1 - Z * Z)) # tanh deriv
    return w1

def derivative_b1(Z, T, Y, W2):
    # return (np.outer(T - Y, W2) * Z *(1 - Z)).sum(axis=0) # sigmoid
    return (np.outer(T - Y, W2) * (1 - Z * Z)).sum(axis=0)
    #return ((T - Y) @ W2.T * (1 - Z * Z)).sum(axis=0)

def cost(T, Y):
    return (T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()

def test_xor():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    T = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 4)
    b1 = np.random.randn(4)
    W2 = np.random.randn(4)
    b2 = np.random.randn(1)
    LL = [] # keep track of likelihoods
    learning_rate = .0005
    l2 = .1
    last_error_rate = None

    for i in range(100000):
        Y, Z = forward(X, W1, b1, W2, b2)
        ll = cost(T, Y)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - T).mean()
        # print error rate if it is different than the last pass
        if er != last_error_rate:
            last_error_rate = er
            print('error rate:', er)
            print('true:', T)
            print('pred:', prediction)
        # if LL and ll < LL[-1]:
        #     print('early exit')
        #     break
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, T, Y) - l2 * W2)
        b2 += learning_rate * (derivative_b2(T, Y) - l2 * b2)
        W1 += learning_rate * (derivative_w1(X, Z, T, Y, W2) - l2 * W1)
        b1 += learning_rate * (derivative_b1(Z, T, Y, W2) - l2 * b1)
        if i % 10000 == 0:
            print (ll)
    print('final classification rate', 1 - np.abs(prediction - T).mean())
    plt.plot(LL)
    plt.show()

def test_donut():
    N = 1000
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(int(N/2)) + R_inner
    theta= 2*np.pi*np.random.random(int(N/2))
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(int(N/2)) + R_outer
    theta= 2*np.pi*np.random.random(int(N/2))
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    T = np.array([0]*(int(N/2)) + [1]*(int(N/2)))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)

    LL = [] # keep track of likelihoods
    learning_rate = .00005
    l2 = 0.2
    last_error_rate = None

    for i in range(160000):
        Y, Z = forward(X, W1, b1, W2, b2)
        ll = cost(T, Y)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        # print error rate if it is different than the last pass
        if er != last_error_rate:
            last_error_rate = er
            print('error rate:', er)
        if LL and ll < LL[-1]:
            print('early exit')
            break
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, T, Y) - l2 * W2)
        b2 += learning_rate * (derivative_b2(T, Y) - l2 * b2)
        W1 += learning_rate * (derivative_w1(X, Z, T, Y, W2) - l2 * W1)
        b1 += learning_rate * (derivative_b1(Z, T, Y, W2) - l2 * b1)
        if i % 10000 == 0:
            print (ll)
    print('final classification rate', 1 - np.abs(prediction - T).mean())
    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    #test_xor()
    test_donut()
