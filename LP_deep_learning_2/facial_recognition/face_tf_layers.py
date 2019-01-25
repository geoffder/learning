import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


class ANN(object):
    '''
    Generate new neural network, taking hidden layer sizes as a list, and
    dropout rate as p_keep
    '''
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, T, Xvalid, Tvalid, lr=1e-4, mu=0.9, decay=0.9, epochs=15,
            batch_sz=100, print_every=50):
        '''
        Takes training data and test data (valid) at once, then trains and
        validates along the way. Modifying hyperparams of learning_rate, mu,
        decay, epochs (iterations = N//batch_sz * epochs), batch_sz and how
        often to validate and print results are passed as optional variables.
        '''
        X = X.astype(np.float32)
        T = T.astype(np.int64)
        Xvalid = Xvalid.astype(np.float32)
        Tvalid = Tvalid.astype(np.int64)

        # initialize hidden layers
        N, D = X.shape
        K = len(set(T))
        self.hidden_layers = []
        M1 = D  # first input layer is the number of features in X
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2)
            self.hidden_layers.append(h)
            M1 = M2  # input layer to next layer is this layer.
        # output layer weights (last hidden layer to K output classes)
        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect params for later use, output weights are first here.
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # set up theano functions and variables
        inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        logits = self.forward(inputs, True)  # logits to feed to the cost func

        # softmax done within the cost function, not at the end of forward
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )
        '''
        Unlike theano, the train_op function is not fed all of the
        variable-function pairs for updating each of the weights (and caches,
        momentum terms etc). Simply use a training function with the objective
        specified (e.g. .minimize(cost)). The feed_dict that will be provided
        to train_op are the inputs to cost (X:inputs and Y:lables))
        '''
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay,
                                             momentum=mu).minimize(cost)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        '''
        Setting up the last tensor equation placeholders to build the graphs
        that will be used for computation. No values, training loop is next!
        '''
        prediction = self.predict(inputs)  # returns labels

        # validation cost will be calculated separately,
        # since nothing will be dropped
        test_logits = self.forward(inputs, False)
        test_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=test_logits,
                labels=labels
            )
        )

        # create a session and initialize the variables within it
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            print("epoch:", i, "n_batches:", n_batches)
            X, T = shuffle(X, T)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Tbatch = T[j*batch_sz:(j*batch_sz+batch_sz)]

                sess.run(train_op, feed_dict={inputs: Xbatch,
                                              labels: Tbatch})

                if j % print_every == 0:
                    c = sess.run(test_cost, feed_dict={inputs: Xvalid,
                                                       labels: Tvalid})
                    p = sess.run(prediction, feed_dict={inputs: Xvalid})
                    costs.append(c)
                    e = error_rate(Tvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c,
                          "error rate:", e)

        plt.plot(costs)
        plt.show()

    def forward(self, X, is_training):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z, is_training)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward(X, False)
        return tf.argmax(pY, 1)


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def classRebalance(X, T):
    '''
    Take data and labels and increase number of samples for under-represented
    classes by duplicating the existing ones.
    '''
    classes = np.unique(T)
    Xlist = [X[T == k] for k in classes]
    Tlist = [T[T == k] for k in classes]
    bigN = np.max([t.shape for t in Tlist])
    # develop a less coarse way that better approximates the classes
    Xlist = [np.concatenate([x]*(bigN//x.shape[0]), axis=0) for x in Xlist]
    Tlist = [np.concatenate([t]*(bigN//t.shape[0]), axis=0) for t in Tlist]

    return np.concatenate(Xlist, axis=0), np.concatenate(Tlist, axis=0)


def trainTestSplit(X, T, ratio=.5):
    X, T = shuffle(X, T)
    N = X.shape[0]
    Xtrain, Ttrain = X[:int(N*ratio)], T[:int(N*ratio)]
    Xtest, Ttest = X[int(N*ratio):], T[int(N*ratio):]
    return Xtrain, Ttrain, Xtest, Ttest


def labelEncode(labels):
    N, K = labels.shape[0], np.unique(labels).shape[0]
    indicator = np.zeros((N, K))
    indicator[:, labels] = 1
    return indicator


def main():
    print('loading in data...')
    df = pd.read_csv('fer2013.csv')
    print('data loaded.')
    print('samples in full dataset:', df['emotion'].values.size)
    maxN = 10000  # number of samples to use from dataset (crashes memory)

    pixels = np.array(
        [str.split(' ') for str in df['pixels'].values[:maxN]]
    ).astype(np.uint8)

    X, T = classRebalance(pixels, df['emotion'].values[:maxN])
    print('X shape:', X.shape, 'T shape:', T.shape)
    print('emotion counts:', [(T == k).sum() for k in np.unique(T)])

    Xtrain, Ttrain_labels, Xtest, Ttest_labels = trainTestSplit(X, T, ratio=.8)
    Ttrain, Ttest = labelEncode(Ttrain_labels), labelEncode(Ttest_labels)
    ann = ANN([300, 300, 300])
    ann.fit(Xtrain, Ttrain, Xtest, Ttest)


if __name__ == '__main__':
    main()
