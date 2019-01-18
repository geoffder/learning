import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from LP_util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


def main():
    Xtrain, Xtest, Ttrain, Ttest = get_normalized_data()
    Ttrain, Ttest = y2indicator(Ttrain), y2indicator(Ttest)

    lr = .00004
    reg = .01

    max_iter = 20
    print_period = 10

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300  # hidden units
    K = 10  # output classes

    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    thX = T.matrix('X')  # placeholders
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')  # updateable
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')

    thZ = T.nnet.relu(thX.dot(W1) + b1)  # more placeholders for calculations
    thY = T.nnet.softmax(thZ.dot(W2) + b2)

    # cost function that will be differentiated
    cost = -(thT*T.log(thY)).sum() + reg*((W1**2).sum() + (b1**2).sum()
                                          + (W2**2).sum() + (b2**2).sum())
    prediction = T.argmax(thY, axis=1)

    update_W1 = W1 - lr*T.grad(cost, W1)  # derivative of cost wrt W1
    update_b1 = b1 - lr*T.grad(cost, b1)
    update_W2 = W2 - lr*T.grad(cost, W2)
    update_b2 = b2 - lr*T.grad(cost, b2)

    train = theano.function(
        inputs=[thX, thT],
        updates=[[W1, update_W1], [b1, update_b1], [W2, update_W2],
                 [b2, update_b2]]
    )
    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction]
    )

    LL = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]

            train(Xbatch, Tbatch)
            if j % print_period == 0:
                cost_val, Ptest = get_prediction(Xtest, Ttest)
                err = error_rate(Ptest, np.argmax(Ttest, axis=1))
                print("cost / err at iteration i=%d, j=%d: %.3f / %.3f"
                      % (i, j, cost_val, err))
                LL.append(cost_val)

    cost_val, Ytest = get_prediction(Xtest, Ttest)
    print("final error rate:", error_rate(Ptest, np.argmax(Ttest, axis=1)))
    plt.plot(LL)
    plt.show()


if __name__ == "__main__":
    main()
