import numpy as np
import matplotlib.pyplot as plt

from LP_mlp import forward, derivative_w1, derivative_w2, derivative_b1, derivative_b2
from LP_util import get_normalized_data, error_rate, cost, y2indicator
from sklearn.utils import shuffle

max_iter = 20
print_period = 10

Xtrain, Xtest, Ttrain, Ttest = get_normalized_data()
lr = .00004
reg = .01

Ttrain = y2indicator(Ttrain)
Ttest = y2indicator(Ttest)

N, D = Xtrain.shape
batch_sz = 500
n_batches = N // batch_sz

M = 300 # hidden units, around the number of components according to PCA
K = Ttrain.shape[1]
W1_0 = np.random.randn(D, M) / np.sqrt(D)
b1_0 = np.zeros(M)
W2_0 = np.random.randn(M, K) / np.sqrt(M)
b2_0 = np.random.randn(K)

# batch gradient descent (no momentum)
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()
LL_batch = []
ER_batch = []
for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Tbatch = Ttrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

        #updates
        W2 -= lr * (derivative_w2(Z, Tbatch, Ybatch) + reg * W2)
        b2 -= lr * (derivative_b2(Tbatch, Ybatch) + reg * b2)
        W1 -= lr * (derivative_w1(Xbatch, Z, Tbatch, Ybatch, W2) + reg * W1)
        b1 -= lr * (derivative_b1(Z, Tbatch, Ybatch, W2) + reg * b1)

        if j % print_period == 0:
            Y, _ = forward(Xtest, W1, b1, W2, b2)
            l = cost(Y, Ttest)
            LL_batch.append(l)
            print("Cost at interation i=%d, j=%d: %.6f" % (i, j, l))

            e = error_rate(Y, np.argmax(Ttest, axis=1))
            ER_batch.append(e)
            print('error rate:', e)

Y, _ = forward(Xtest, W1, b1, W2, b2)
print('Final error rate (no momentum):', error_rate(Y, np.argmax(Ttest, axis=1)))

# batch GD with momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()

#momentum terms
mu = .9 # velocity weight constant
dW2 = 0
db2 = 0
dW1 = 0
db1 = 0

LL_momentum = []
ER_momentum = []
for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Tbatch = Ttrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

        # gradients
        gW2 = derivative_w2(Z, Tbatch, Ybatch) + reg * W2
        gb2 = derivative_b2(Tbatch, Ybatch) + reg * b2
        gW1 = derivative_w1(Xbatch, Z, Tbatch, Ybatch, W2) + reg * W1
        gb1 = derivative_b1(Z, Tbatch, Ybatch, W2) + reg * b1

        # update velocities
        dW2 = mu * dW2 - lr * gW2
        db2 = mu * db2 - lr * gb2
        dW1 = mu * dW1 - lr * gW1
        db1 = mu * db1 - lr * gb1

        # update weights
        W2 += dW2
        b2 += db2
        W1 += dW1
        b1 += db1

        if j % print_period == 0:
            Y, _ = forward(Xtest, W1, b1, W2, b2)
            l = cost(Y, Ttest)
            LL_momentum.append(l)
            print("Cost at interation i=%d, j=%d: %.6f" % (i, j, l))

            e = error_rate(Y, np.argmax(Ttest, axis=1))
            ER_momentum.append(e)
            print('error rate:', e)

Y, _ = forward(Xtest, W1, b1, W2, b2)
print('Final error rate (momentum):', error_rate(Y, np.argmax(Ttest, axis=1)))

# batch GD with nesterov momentum
W1 = W1_0.copy()
b1 = b1_0.copy()
W2 = W2_0.copy()
b2 = b2_0.copy()

#momentum terms
mu = .9 # velocity weight constant
dW2 = 0
db2 = 0
dW1 = 0
db1 = 0

LL_nesterov = []
ER_nesterov = []
for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Tbatch = Ttrain[j*batch_sz:(j*batch_sz + batch_sz)]
        Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

        # gradients
        gW2 = derivative_w2(Z, Tbatch, Ybatch) + reg * W2
        gb2 = derivative_b2(Tbatch, Ybatch) + reg * b2
        gW1 = derivative_w1(Xbatch, Z, Tbatch, Ybatch, W2) + reg * W1
        gb1 = derivative_b1(Z, Tbatch, Ybatch, W2) + reg * b1

        # update velocities
        dW2 = mu*dW2 - lr*gW2
        db2 = mu*db2 - lr*gb2
        dW1 = mu*dW1 - lr*gW1
        db1 = mu*db1 - lr*gb1

        # update weights
        W2 += mu*dW2 - lr*gW2
        b2 += mu*db2 - lr*gb2
        W1 += mu*dW1 - lr*gW1
        b1 += mu*db1 - lr*gb1

        if j % print_period == 0:
            Y, _ = forward(Xtest, W1, b1, W2, b2)
            l = cost(Y, Ttest)
            LL_nesterov.append(l)
            print("Cost at interation i=%d, j=%d: %.6f" % (i, j, l))

            e = error_rate(Y, np.argmax(Ttest, axis=1))
            ER_nesterov.append(e)
            print('error rate:', e)

Y, _ = forward(Xtest, W1, b1, W2, b2)
print('Final error rate (nesterov):', error_rate(Y, np.argmax(Ttest, axis=1)))

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(LL_batch, label='no momentum')
ax1.plot(LL_momentum, label='regular momentum')
ax1.plot(LL_nesterov, label='nesterov momentum')
ax1.legend()
ax1.set_title('cost')
ax2.plot(ER_batch, label='no momentum')
ax2.plot(ER_momentum, label='regular momentum')
ax2.plot(ER_nesterov, label='nesterov momentum')
ax2.legend()
ax2.set_title('error rate')
plt.show()
