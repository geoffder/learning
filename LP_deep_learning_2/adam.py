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

    Ttrain, Ttest = y2indicator(Ttrain), y2indicator(Ttest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300  # hidden units
    K = 10  # output classes

    # use the same initial weights for adam and RMSprop+momentum
    W1_0 = np.random.randn(D, M) / np.sqrt(D)
    b1_0 = np.zeros(M)
    W2_0 = np.random.randn(M, K) / np.sqrt(M)
    b2_0 = np.zeros(K)

    # intial learning rate and decay
    lr0 = .001  # don't go too high, or NaNs will result
    beta1 = .9  # decay constants for first and second moment
    beta2 = .999
    eps = 1e-8

    # 1. Adam
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    # 1st moment
    mW1 = 0
    mb1 = 0
    mW2 = 0
    mb2 = 0

    # 2nd moment
    vW1 = 0
    vb1 = 0
    vW2 = 0
    vb2 = 0

    loss_adam = []
    err_adam = []
    t = 1  # by convention. Otherwise, first correction is 0 (div by 0)
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # gradients
            gW2 = derivative_w2(Z, Tbatch, Ybatch) + reg*W2
            gb2 = derivative_b2(Tbatch, Ybatch) + reg*b2
            gW1 = derivative_w1(Xbatch, Z, Tbatch, Ybatch, W2) + reg*W1
            gb1 = derivative_b1(Z, Tbatch, Ybatch, W2) + reg*b1
            # new first moment (m)
            # exponentially smoothed avg of the gradients
            mW1 = beta1*mW1 + (1 - beta1)*gW1  # add fraction of gradient
            mb2 = beta1*mb1 + (1 - beta1)*mb1
            mW2 = beta1*mW2 + (1 - beta1)*gW2
            mb2 = beta1*mb2 + (1 - beta1)*gb2
            # new second moment (v)
            # exponentially smoothed avg of the squared gradients
            vW1 = beta2*vW1 + (1 - beta2)*gW1**2  # add fraction of gradient^2
            vb2 = beta2*vb1 + (1 - beta2)*mb1**2
            vW2 = beta2*vW2 + (1 - beta2)*gW2**2
            vb2 = beta2*vb2 + (1 - beta2)*gb2**2
            # bias correction
            correction1 = 1 - beta1**t
            hat_mW1 = mW1/correction1  # boost first updates (against 0 bias)
            hat_mb1 = mb1/correction1
            hat_mW2 = mW2/correction1
            hat_mb2 = mb2/correction1
            correction2 = 1 - beta2**t
            hat_vW1 = vW1/correction2  # boost first updates (against 0 bias)
            hat_vb1 = vb1/correction2
            hat_vW2 = vW2/correction2
            hat_vb2 = vb2/correction2
            t += 1  # update t (each learning step)
            # update weights
            W1 -= lr0 * hat_mW1 / (np.sqrt(hat_vW1) + eps)
            b1 -= lr0 * hat_mb1 / (np.sqrt(hat_vb1) + eps)
            W2 -= lr0 * hat_mW2 / (np.sqrt(hat_vW2) + eps)
            b2 -= lr0 * hat_mb2 / (np.sqrt(hat_vb2) + eps)

            if j % print_period == 0:
                # calculate Log-Likelihood
                Y, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(Y, Ttest)
                loss_adam.append(ll)
                print('cost at iteration i=%d, j=%d %.6f' % (i, j, ll))

                err = error_rate(Y, np.argmax(Ttest, axis=1))
                err_adam.append(err)
                print('error rate:', err)

    Y, _ = forward(Xtest, W1, b1, W2, b2)
    print('final error rate (adam):', error_rate(Y, np.argmax(Ttest, axis=1)))

    # 2. RMSprop + momentum
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    cache_W1 = 1  # updates more comparable with adam
    cache_b1 = 1
    cache_W2 = 1
    cache_b2 = 1

    dW1 = 0
    db1 = 0
    dW2 = 0
    db2 = 0

    loss_rmsmom = []
    err_rmsmom = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Ybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # gradients
            gW2 = derivative_w2(Z, Tbatch, Ybatch) + reg*W2
            gb2 = derivative_b2(Tbatch, Ybatch) + reg*b2
            gW1 = derivative_w1(Xbatch, Z, Tbatch, Ybatch, W2) + reg*W1
            gb1 = derivative_b1(Z, Tbatch, Ybatch, W2) + reg*b1
            # RMSprop cache (less learning as more learning occurs)
            cache_W1 = beta2*cache_W1 + (1 - beta2)*gW1*gW1
            cache_b1 = beta2*cache_b1 + (1 - beta2)*gb1*gb1
            cache_W2 = beta2*cache_W2 + (1 - beta2)*gW2*gW2
            cache_b2 = beta2*cache_b2 + (1 - beta2)*gb2*gb2
            # velocity term (momentum)
            # dW1 = dW1*beta1 + lr0*gW1 # this is plain velocity..
            # db1 = db1*beta1 + lr0*gb1
            # dW2 = dW2*beta1 + lr0*gW2
            # db2 = db2*beta1 + lr0*db2
            # LP added (1- beta1) discussed in adam lecture, he says it makes
            # the comparison more fair. (velo only topped up)
            # add some gradient divided by learning history cache
            dW1 = dW1*beta1 + (1-beta1)*lr0*gW1/(np.sqrt(cache_W1)+eps)
            db1 = db1*beta1 + (1-beta1)*lr0*gb1/(np.sqrt(cache_b1)+eps)
            dW2 = dW2*beta1 + (1-beta1)*lr0*gW2/(np.sqrt(cache_W2)+eps)
            db2 = db2*beta1 + (1-beta1)*lr0*db2/(np.sqrt(cache_b2)+eps)
            # subtract accumlated velocity
            W2 -= dW2
            b2 -= db2
            W1 -= dW1
            b1 -= db1

            if j % print_period == 0:
                # calculate LL
                Y, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(Y, Ttest)
                loss_rmsmom.append(ll)
                print('cost at iteration i=%d, j=%d %.6f' % (i, j, ll))

                err = error_rate(Y, np.argmax(Ttest, axis=1))
                err_rmsmom.append(err)
                print('error rate:', err)

    Y, _ = forward(Xtest, W1, b1, W2, b2)
    print('final error rate (RMSprop+mom):',
          error_rate(Y, np.argmax(Ttest, axis=1)))

    plt.plot(loss_adam, label='adam')
    plt.plot(loss_rmsmom, label='RMSprop+momentum')
    plt.legend()
    plt.show()


main()
