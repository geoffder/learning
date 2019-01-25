import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


class HiddenLayer(object):
    '''
    Generate new hidden layer with input size and number of nodes as args.
    This is now a batch norm layer.
    '''
    def __init__(self, M1, M2):
        self.M1 = M1  # input layers nodes
        self.M2 = M2  # this layers nodes
        W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        self.W = tf.Variable(W.astype(np.float32))
        self.params = [self.W]
        '''batch norm variables'''
        self.decay = tf.Variable(.9, dtype=tf.float32, trainable=False)
        # length M2, since batch norm is concerned with values immediately
        # before activation functions (X @ W has shape M2)
        self.running_mean = tf.Variable(np.zeros(M2),
                                        dtype=tf.float32, trainable=False)
        self.running_var = tf.Variable(np.zeros(M2),
                                       dtype=tf.float32, trainable=False)
        self.beta = tf.Variable(0.0, dtype=tf.float32)
        self.gamma = tf.Variable(1.0, dtype=tf.float32)
        self.epsilon = tf.Variable(.0001, dtype=tf.float32)

    def forward(self, X, is_training):
        '''Calculate values for next layer with ReLU activation'''
        a = tf.matmul(X, self.W)
        if is_training:
            batch_mean, batch_var = tf.nn.moments(a, [0])
            update_rn_mean = tf.assign(
                self.running_mean,
                tf.add(tf.multiply(self.running_mean, self.decay),
                       tf.multiply(batch_mean,  1 - self.decay))
            )
            update_rn_var = tf.assign(
                self.running_var,
                tf.add(tf.multiply(self.running_var, self.decay),
                       tf.multiply(batch_var, 1 - self.decay))
            )
            # ensures these update functions and all that they depend, which
            # includes batch_mean and batch_var before the following code block
            with tf.control_dependencies([update_rn_mean, update_rn_var]):
                a_norm = tf.nn.batch_normalization(
                    a, batch_mean, batch_var,
                    self.beta, self.gamma, self.epsilon
                )
        else:
            a_norm = tf.nn.batch_normalization(
                a, self.running_mean, self.running_var,
                self.beta, self.gamma, self.epsilon
            )
        return tf.nn.relu(a_norm)


class ANN(object):
    '''
    Generate new neural network, taking hidden layer sizes as a list, and
    dropout rate as p_keep
    '''
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep

    def fit(self, X, T, Xvalid, Tvalid, lr=1e-4, mu=0.9, decay=0.9, epochs=40,
            batch_sz=200, print_every=50):
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
        K = np.unique(T).shape[0]
        self.hidden_layers = []
        M1 = D  # first input layer is the number of features in X
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2)
            self.hidden_layers.append(h)
            M1 = M2  # input layer to next layer is this layer.
        # output layer weights (last hidden layer to K output classes)
        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        self.W = tf.Variable(W.astype(np.float32))

        # collect params for later use, output weights are first here.
        self.params = [self.W]
        for h in self.hidden_layers:
            self.params += h.params

        # set up theano functions and variables
        inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        logits = self.forward_dropout(inputs, True)

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
        # train_op = tf.train.RMSPropOptimizer(lr, decay=decay,
        #                                      momentum=mu).minimize(cost)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        '''
        Setting up the last tensor equation placeholders to build the graphs
        that will be used for computation. No values, training loop is next!
        '''
        prediction = self.predict(inputs)  # returns labels

        # validation cost will be calculated separately,
        # since nothing will be dropped
        test_logits = self.forward_test_dropout(inputs, False)
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
        return tf.matmul(Z, self.W)

    def predict(self, X):
        pY = self.forward(X, False)
        return tf.argmax(pY, 1)

    def forward_dropout(self, X, is_training):
        # tf.nn.dropout scales inputs by 1/p_keep
        # therefore, during test time, we don't have to scale anything
        Z = X
        '''
        Here is the dropout implementation. Tensorflow does the masking for us.

        Inputs (X) dropout is done first, outside the loop. Then dropout is
        performed on the outputs, before they become inputs to the next layer.
        This is a bit different from the theano code. Haven't thought about
        why he did this hard enough yet. I believe the outcome is the same, but
        the theano implementation is cleaner since it is all within the loop.
        '''
        Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
            Z = h.forward(Z, is_training)
            Z = tf.nn.dropout(Z, p)
        return tf.matmul(Z, self.W)  # logits of final layer.

    def forward_test_dropout(self, X, is_training):
        '''
        Note that inputs aren't scaled here, the dropout function already
        took care of that during training (see LP's note above).
        '''
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z, is_training)
        return tf.matmul(Z, self.W)

    def predict_dropout(self, X):
        pY = self.forward_test_dropout(X, False)
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
    maxN = 20000  # number of samples to use from dataset (crashes memory)

    pixels = np.array(
        [str.split(' ') for str in df['pixels'].values[:maxN]]
    ).astype(np.uint8)

    X, T = classRebalance(pixels, df['emotion'].values[:maxN])
    print('X shape:', X.shape, 'T shape:', T.shape)
    print('emotion counts:', [(T == k).sum() for k in np.unique(T)])

    # Xtrain, Ttrain_labels, Xtest, Ttest_labels = trainTestSplit(X, T,
    #                                                             ratio=.8)
    # Ttrain, Ttest = labelEncode(Ttrain_labels), labelEncode(Ttest_labels)
    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T, ratio=.8)
    ann = ANN([1000, 1000, 500, 500, 300, 100],
              [0.8, 0.5, 0.5, .5, .5, .5, .5])
    # ann = ANN([500, 500, 300, 100, 100], [0.8, 0.5, 0.5, .5, .5, .5])
    ann.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=100)


if __name__ == '__main__':
    main()
