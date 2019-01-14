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

    maxNodes, maxLayers = 400, 4
    # starting hyperparams
    numLayers = 2  # np.random.randint(1, maxLayers+1, 1)
    sizes = [int(maxNodes/numLayers) for i in range(numLayers)]
    r = -5  # np.random.uniform(-8, -2)  # lr = 10**r
    l2 = .5  # np.random.uniform(0, 1)

    # loop through all possible hyperparam settings
    best_validation_rate = 0
    best_sizes, best_lr, best_l2 = [None for i in range(3)]

    for i in range(30):
        if 1 < numLayers < maxLayers:
            new_numLayers = int(numLayers + np.random.randint(-1, 2, size=1))
        elif numLayers == 1:
            new_numLayers = int(numLayers + np.random.randint(0, 2, size=1))
        else:
            new_numLayers = int(numLayers + np.random.randint(-1, 1, size=1))
        sizes = [int(maxNodes/new_numLayers) for i in range(new_numLayers)]
        new_r = np.random.uniform(r-2, r+2, size=1)
        lr = 10**new_r
        new_l2 = np.abs(np.random.uniform(l2-.2, l2+.2))

        model = ANN(sizes)
        model.fit(Xtrain, Ytrain, learning_rate=lr, reg=new_l2, mu=.99,
                  epochs=600, show_fig=False)
        validation_acc = model.score(Xtest, Ytest)
        train_acc = model.score(Xtrain, Ytrain)
        print(
         'validation acc: %.3f; train acc: %.3f; settings: %s, %s, %s'
         % (validation_acc, train_acc, sizes, lr, new_l2))
        if validation_acc > best_validation_rate:
            best_validation_rate = validation_acc
            numLayers, r, l2 = new_numLayers, new_r, new_l2
            best_sizes, best_lr, best_l2 = sizes, lr, l2

    print('Best validation_accuracy:', best_validation_rate)
    print('Best settings:')
    print('hidden_layer_sizes:', best_sizes)
    print('learning_rate:', best_lr)
    print('l2:', best_l2)


random_search()
