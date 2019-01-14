from LP_theano_ann import ANN
from LP_util import get_spiral
from sklearn.utils import shuffle
import numpy as np


def random_search():
    '''
    Using some of LPs methods, see other version for my original attempt.
    Similar performance, across all random search scripts. I find that the
    sampling doesn't give as much benefit as I thought vs simple and LP vers.
    All are capable of finding high validation with only 300 epochs.
    '''
    # get the data and split into train/test sets
    X, Y = get_spiral()
    X, Y = shuffle(X, Y)
    Ntrain = int(.7*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # starting hyperparams
    best_nHidden = 2
    best_M = 20
    best_lr = -5
    best_l2 = .5

    # loop through all possible hyperparam settings
    best_validation_rate = 0
    for i in range(15):
        for j in range(2):
            nHidden = max(1, best_nHidden + np.random.randint(-1, 2))
            M = max(10, best_M + np.random.randint(-1, 2)*10)
            log_lr = np.random.uniform(best_lr-1, best_lr+1, size=1)
            log_l2 = np.random.uniform(best_l2-1, best_l2+1, size=1)

            sizes = [M]*nHidden
            model = ANN(sizes)
            model.fit(Xtrain, Ytrain, learning_rate=10**log_lr, reg=10**log_l2,
                      mu=.99, epochs=300, show_fig=False)
            validation_acc = model.score(Xtest, Ytest)
            train_acc = model.score(Xtrain, Ytrain)
            print(
             'validation acc: %.3f; train acc: %.3f; settings: %s, %s, %s'
             % (validation_acc, train_acc, sizes, log_lr, log_l2))
            if validation_acc > best_validation_rate:
                best_validation_rate = validation_acc
                next_M, next_nHidden = M, nHidden
                next_lr, next_l2 = log_lr, log_l2

        best_M, best_nHidden,  = next_M, next_nHidden
        best_lr, best_l2 = next_lr, next_l2

    print('Best validation_accuracy:', best_validation_rate)
    print('Best settings:')
    print('hidden_layer_sizes:', [M]*best_nHidden)
    print('learning_rate:', best_lr)
    print('l2:', best_l2)


random_search()
