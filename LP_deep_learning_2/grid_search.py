from LP_theano_ann import ANN
from LP_util import get_spiral
from sklearn.utils import shuffle


def grid_search():
    # get the data and split into train/test sets
    X, Y = get_spiral()
    X, Y = shuffle(X, Y)
    Ntrain = int(.7*len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # hyper parameters to try
    hidden_layer_sizes = [
        [300],
        [100, 100],
        [50, 50, 50]
    ]
    learning_rates = [10**-(r+2) for r in range(3)]
    l2_penalties = [0, .1, 1]

    # loop through all possible hyperparam settings
    best_validation_rate = 0
    best_sizes, best_lr, best_l2 = [None for i in range(3)]

    for sizes in hidden_layer_sizes:
        for lr in learning_rates:
            for l2 in l2_penalties:
                model = ANN(sizes)
                model.fit(Xtrain, Ytrain, learning_rate=lr, reg=l2, mu=.99,
                          epochs=300, show_fig=False)
                validation_acc = model.score(Xtest, Ytest)
                train_acc = model.score(Xtrain, Ytrain)
                print(
                 'validation acc: %.3f; train acc: %.3f; settings: %s, %s, %s'
                 % (validation_acc, train_acc, sizes, lr, l2))
                if validation_acc > best_validation_rate:
                    best_validation_rate = validation_acc
                    best_sizes, best_lr, best_l2 = sizes, lr, l2
    print('Best validation_accuracy:', best_validation_rate)
    print('Best settings:')
    print('hidden_layer_sizes:', best_sizes)
    print('learning_rate:', best_lr)
    print('l2:', best_l2)


grid_search()
