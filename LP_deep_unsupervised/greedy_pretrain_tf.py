import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.utils import shuffle
from LP_util import getKaggleMNIST


class AutoEncoder(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def fit(self, X, lr=1e-4, epochs=40, batch_sz=200, print_every=50):

        N, D = X.shape

        # initialize parameters
        W_hidden = np.random.randn(D, self.nodes) * np.sqrt(2.0 / D)
        b_hidden = np.zeros(self.nodes)
        W_out = np.random.randn(self.nodes, D) * np.sqrt(2.0 / self.nodes)
        b_out = np.zeros(D)
        self.W_hidden = tf.Variable(W_hidden.astype(np.float32))
        self.b_hidden = tf.Variable(b_hidden.astype(np.float32))
        self.W_out = tf.Variable(W_out.astype(np.float32))
        self.b_out = tf.Variable(b_out.astype(np.float32))
        self.params = [self.W_hidden, self.b_hidden,
                       self.W_out, self.b_out]

        inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        reconstruction = self.forward(inputs)

        cost = tf.reduce_mean(
            tf.losses.mean_squared_error(inputs, reconstruction))

        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        # create a session and initialize the variables within it
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        n_batches = N // batch_sz
        for i in range(epochs):
            print("epoch:", i, "n_batches:", n_batches)
            X = shuffle(X)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

                c, _ = sess.run([cost, train_op], feed_dict={inputs: Xbatch})

                if j % print_every == 0:
                    print("cost: %f" % (c))

        return sess.run(self.get_hidden(X)), sess.run(self.W_hidden)

    def forward(self, X):
        '''Full Encode / Decode'''
        X = tf.nn.dropout(X, .5)
        Z = tf.matmul(X, self.W_hidden) + self.b_hidden
        out = tf.nn.sigmoid(tf.matmul(Z, self.W_out) + self.b_out)
        return out

    def get_hidden(self, X):
        '''Return hidden representation (input to next autoencoder)'''
        return tf.matmul(X, self.W_hidden) + self.b_hidden


class HiddenLayer(object):
    '''
    Generate new hidden layer with input size and number of nodes as args.
    This is now a batch norm layer.
    '''
    def __init__(self, M1, M2, pre_weight=None):
        self.M1 = M1  # input layers nodes
        self.M2 = M2  # this layers nodes
        if pre_weight is None:
            W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        else:
            W = pre_weight
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]
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
        self.epsilon = tf.Variable(.0000001, dtype=tf.float32)

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

    def __init__(self, hidden_layer_sizes, dropout_rates, weights):
        self.nodes = hidden_layer_sizes
        self.dropout_rates = dropout_rates
        self.pre_weights = weights

    def fit(self, X, T, Xvalid, Tvalid, lr=1e-4, mu=0.9, decay=0.9, epochs=15,
            batch_sz=100, print_every=50):

        X = X.astype(np.float32)
        T = T.astype(np.int64)
        Xvalid = Xvalid.astype(np.float32)
        Tvalid = Tvalid.astype(np.int64)

        # initialize hidden layers
        N, D = X.shape
        K = len(set(T))
        self.hidden_layers = []
        M1 = D  # first input layer is the number of features in X
        for M2, W in zip(self.nodes, self.pre_weights):
            h = HiddenLayer(M1, M2, pre_weight=W)
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
        logits = self.forward_dropout(inputs, True)  # fed to cost function

        # softmax done within the cost function, not at the end of forward
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )

        # train_op = tf.train.RMSPropOptimizer(lr, decay=decay,
        #                                      momentum=mu).minimize(cost)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        prediction = self.predict(inputs)  # returns labels

        # validation cost will be calculated separately,
        # since nothing will be dropped
        test_logits = self.forward_dropout(inputs, False)
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
        train_costs, test_costs, train_errs, test_errs = [[] for _ in range(4)]
        for i in range(epochs):
            print("epoch:", i, "n_batches:", n_batches)
            X, T = shuffle(X, T)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Tbatch = T[j*batch_sz:(j*batch_sz+batch_sz)]

                sess.run(train_op, feed_dict={inputs: Xbatch, labels: Tbatch})

                if j % print_every == 0:
                    # train set
                    trc = sess.run(cost, feed_dict={inputs: Xvalid,
                                                    labels: Tvalid})
                    train_costs.append(trc)
                    p = sess.run(prediction, feed_dict={inputs: Xvalid})
                    train_e = error_rate(Tvalid, p)
                    train_errs.append(train_e)
                    # test set
                    c = sess.run(test_cost, feed_dict={inputs: Xvalid,
                                                       labels: Tvalid})
                    test_costs.append(c)
                    p = sess.run(prediction, feed_dict={inputs: Xvalid})
                    test_e = error_rate(Tvalid, p)
                    test_errs.append(test_e)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c,
                          "error rate:", test_e)

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(train_costs, label='training cost')
        axes[0].plot(test_costs, label='validation cost')
        axes[0].legend()
        axes[1].plot(train_errs, label='training error rate')
        axes[1].plot(test_errs, label='validation error rate')
        axes[1].legend()
        fig.tight_layout()
        plt.show()

    def forward_dropout(self, X, is_training):
        # tf.nn.dropout scales inputs by 1/p_keep
        # therefore, during test time, we don't have to scale anything
        Z = X
        if is_training:
            Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
            Z = h.forward(Z, is_training)
            if is_training:
                Z = tf.nn.dropout(Z, p)
        return tf.matmul(Z, self.W) + self.b  # logits of final layer.

    def predict(self, X):
        pY = self.forward_dropout(X, False)
        return tf.argmax(pY, 1)


def error_rate(p, t):
    return np.mean(p != t)


def main():
    Xtrain, Ttrain, Xtest, Ttest, = getKaggleMNIST()

    hidden_layer_sizes = [500, 300, 100]

    ae_stack = [AutoEncoder(nodes) for nodes in hidden_layer_sizes]
    ae_weights = [[] for _ in range(len(hidden_layer_sizes))]
    input = Xtrain
    for i, ae in enumerate(ae_stack):
        print('#### BEGIN TRAINING FOR AUTOENCODER %s ####' % i)
        input, ae_weights[i] = ae.fit(input, lr=1e-3, epochs=10)

    # now train a network with these pre-trained weights
    dropout_rates = [.8, .5, .5, .5]
    ann = ANN(hidden_layer_sizes, dropout_rates, ae_weights)
    print('#### BEGIN FINE TUNING AND CLASSIFICATION ####')
    ann.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=30)


if __name__ == '__main__':
    main()
