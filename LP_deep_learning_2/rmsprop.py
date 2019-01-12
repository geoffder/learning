import numpy as np
import matplotlib.pyplot as plt

from LP_util import get_normalized_data, error_rate, cost, y2indicator
from LP_mlp import forward, derivative_w2, derivative_w1
from LP_mlp import derivative_b2, derivative_b1


def main():
    max_iter = 20
    print_period = 10

    Xtrain, Xtest, Ttrain, Ttest = get_normalized_data()
    reg = .01
    lr = .0001

    Ttrain, Ttest = y2indicator(Ttrain), y2indicator(Ttest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300  # hidden units
    K = 10  # output classes

    # 1. constant learning rate
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(K)
    b2 = np.zeros(K)

    LL_batch = []
    CR_batch = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            W2 -= lr * (derivative_w2(Z, Tbatch, Ybatch) + reg*W2)
            b2 -= lr * (derivative_b2(Tbatch, Ybatch) + reg*b2)
            W1 -= lr * (derivative_w1(Xbatch, Z, Tbatch, Ybatch, W2) + reg*W1)
            b1 -= lr * (derivative_b1(Z, Tbatch, Ybatch, W2) + reg*b1)

            if j % print_period == 0:
                # calculate LL
                Y, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(Y, Ttest)
                LL_batch.append(ll)
                print('cost at iteration i=%d, j=%d %.6f' % (i, j, ll))

                err = error_rate(Y, np.argmax(Ttest, axis=1))
                CR_batch.append(err)
                print('error rate:', err)

    Y, _ = forward(Xtest, W1, b1, W2, b2)
    print('final error rate (batch):', error_rate(Y, np.argmax(Ttest, axis=1)))

    # 1. RMSprop
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(K)
    b2 = np.zeros(K)

    # intial learning rate and decay
    lr0 = .0005  # don't go too high, or NaNs will result
    decay_rate = .999
    eps = .000001  # epsilon (constant, see notes)
    cache_W1 = 0
    cache_b1 = 0
    cache_W2 = 0
    cache_b2 = 0

    LL_rms = []
    CR_rms = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # updates
            gW2 = derivative_w2(Z, Tbatch, Ybatch) + reg*W2  # gradient
            # replace fraction of cache with new gradient
            cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
            # divide gradient by sqrt of cache
            # (greater learning history -> less new learning)
            W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)

            gb2 = derivative_b2(Tbatch, Ybatch) + reg*b2
            cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
            b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)

            gW1 = derivative_w1(Xbatch, Z, Tbatch, Ybatch, W2) + reg*W1
            cache_W1 = decay_rate * cache_W1 + (1 - decay_rate)*gW1*gW1
            W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)

            gb1 = derivative_b1(Z, Tbatch, Ybatch, W2) + reg*b1
            cache_b1 = decay_rate * cache_b1 + (1 - decay_rate)*gb1*gb1
            b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)

            if j % print_period == 0:
                # calculate LL
                Y, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(Y, Ttest)
                LL_rms.append(ll)
                print('cost at iteration i=%d, j=%d %.6f' % (i, j, ll))

                err = error_rate(Y, np.argmax(Ttest, axis=1))
                CR_rms.append(err)
                print('error rate:', err)

    Y, _ = forward(Xtest, W1, b1, W2, b2)
    print('final error rate (RMSprop):',
          error_rate(Y, np.argmax(Ttest, axis=1)))

    plt.plot(LL_batch, label='batch')
    plt.plot(LL_rms, label='RMSprop')
    plt.legend()
    plt.show()


main()
