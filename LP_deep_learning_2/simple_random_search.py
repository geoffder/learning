from LP_theano_ann import ANN
from LP_util import get_spiral
from sklearn.utils import shuffle
import numpy as np


def random_search():
    '''updated from original to use some of LPs lines.'''
    # get the data and split into train/test sets
    X, Y = get_spiral()
    X, Y = shuffle(X, Y)
    Ntrain = int(.7*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # starting hyperparams
    nHidden = 2  # np.random.randint(1, 4, 1)
    M = 20
    nHidden = 2
    log_lr = -4  # np.random.uniform(-8, -2)  # lr = 10**log_lr
    log_l2 = -2

    # loop through all possible hyperparam settings
    best_validation_rate = 0
    best_sizes, best_lr, best_l2 = [None for i in range(3)]

    for i in range(30):
        sizes = [M]*nHidden
        model = ANN(sizes)
        model.fit(Xtrain, Ytrain, learning_rate=10**log_lr, reg=10**log_l2,
                  mu=.99, epochs=300, show_fig=False)
        validation_acc = model.score(Xtest, Ytest)
        train_acc = model.score(Xtrain, Ytrain)
        print(
         'validation acc: %.3f; train acc: %.3f; settings: %s, %s, %s'
         % (validation_acc, train_acc, sizes, 10**log_lr, 10**log_l2))
        if validation_acc > best_validation_rate:
            best_validation_rate = validation_acc
            best_M, best_nHidden, best_lr, best_l2 = M, nHidden, log_lr, log_l2

        nHidden = max(1, best_nHidden + np.random.randint(-1, 2))
        M = max(10, best_M + np.random.randint(-1, 2)*10)
        log_lr = np.random.uniform(best_lr-1, best_lr+1, size=1)
        log_l2 = np.random.uniform(best_l2-1, best_l2+1, size=1)

    print('Best validation_accuracy:', best_validation_rate)
    print('Best settings:')
    print('hidden_layer_sizes:', [M]*best_nHidden)
    print('learning_rate:', 10**best_lr)
    print('l2:', 10**best_l2)


random_search()
