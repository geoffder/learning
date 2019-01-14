from LP_theano_ann import ANN
from LP_util import get_spiral
from sklearn.utils import shuffle
import numpy as np


def random_search():
    # get the data and split into train/test sets
    X, Y = get_spiral()
    X, Y = shuffle(X, Y)
    Ntrain = int(.7*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # starting hyperparameters
    M = 20
    nHidden = 2
    log_lr = -4
    log_l2 = -2  # since we always want it to be positive
    max_tries = 30

    best_validation_rate = 0
    best_M, best_nHidden, best_lr, best_l2 = [None for _ in range(4)]

    for _ in range(max_tries):
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
            best_M, best_nHidden, best_lr, best_l2 = M, nHidden, log_lr, log_l2

        # select new hyperparameters
        nHidden = best_nHidden + np.random.randint(-1, 2)  # -1, 0, or 1
        nHidden = max(1, nHidden)  # less chance to change this way than if..
        # not sure which way is better, his or mine.
        # incr/decr by 10 units at a time. Nodes/layers being split is def
        # better than my lazy implentation
        M = best_M + np.random.randint(-1, 2)*10
        M = max(10, M)  # lower bounds. This max way is nice and consistent
        log_lr = best_lr + np.random.randint(-1, 2)
        # sticking to logs as well for l2 is cleaner than mine for sure
        log_l2 = best_l2 + np.random.randint(-1, 2)

    print('Best validation_accuracy:', best_validation_rate)
    print('Best settings:')
    print('node number (M):', best_M)
    print('number of layers:', best_nHidden)
    print('learning_rate:', 10**best_lr)
    print('l2:', 10**best_l2)


random_search()
