import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from LP_util import get_poetry_classifier_data
from sklearn.utils import shuffle
from os import getcwd

"""
Classification of lines of poetry as being written by Frost or Poe based on
parts-of-speech tag sequences. Tensorflow implementation using Gated Recurrent
Unit, rather than a Simple/Elman unit.

-> v2: Now with validation set testing in the fit() function, instead of just
training on all the data.
"""


def init_weight(M1, M2):
    """
    The weights being small is really important, my pytorch implementation did
    not work until I remembered to divide the randn weights like so.
    """
    return np.random.randn(M1, M2) / np.sqrt(M1 + M2)


class GRU(object):
    def __init__(self, M1, M2):
        self.M1 = M1  # input size
        self.M2 = M2  # hidden layer size
        self.build()

    def build(self):
        # input weight (transforms X before entering the hidden recurrence)
        self.Wxh = tf.Variable(
            init_weight(self.M1, self.M2).astype(np.float32))
        # hidden weight and bias (recurrent)
        self.Whh = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.bh = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # update gate weights
        self.Wxz = tf.Variable(
            init_weight(self.M1, self.M2).astype(np.float32))
        self.Whz = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.bz = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # reset gate weights
        self.Wxr = tf.Variable(
            init_weight(self.M1, self.M2).astype(np.float32))
        self.Whr = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.br = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # initial hidden repesentation
        self.h0 = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        self.params = [self.Wxh, self.Whh, self.bh, self.h0, self.Wxz,
                       self.Whz, self.bz, self.Wxr, self.Whr, self.br]

    def recurrence(self, last, new):
        # reshape recurrent inputs
        last = tf.reshape(last, (1, self.M2))
        new = tf.reshape(new, (1, -1))
        # calculate gates
        reset = tf.sigmoid(
            tf.matmul(new, self.Wxr) + tf.matmul(last, self.Whr) + self.br)
        update = tf.sigmoid(
            tf.matmul(new, self.Wxz) + tf.matmul(last, self.Whz) + self.bz)
        # update hidden representation
        h_hat = tf.nn.relu(
            tf.matmul(new, self.Wxh)
            + (tf.matmul(reset*last, self.Whh) + self.bh))
        hidden = h_hat*update + last*(1-update)
        return tf.reshape(hidden, (self.M2,))

    def scanner(self, X):
        """
        The recurrent loop of this RNN layer. h0 is the "last" arg
        for the first element of the input. We are initializing at zero.
        """
        self.scan = tf.scan(
            fn=self.recurrence,  # run this on each element of the input
            elems=X,
            initializer=self.h0,  # zeros
        )
        return self.scan


class HiddenLayer(object):
    """
    Simple layer with optional non-linearity, and bias. For example, can be
    used to get logits by setting activation=None.
    """
    def __init__(self, M1, M2, bias=True, activation=tf.nn.relu):
        self.M1 = M1  # input size
        self.M2 = M2  # hidden layer size
        self.bias = bias
        self.activation = activation
        self.build()

    def build(self):
        self.W = tf.Variable(init_weight(self.M1, self.M2).astype(np.float32))
        if self.bias:
            self.b = tf.Variable(np.zeros(self.M2, dtype=np.float32))
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def forward(self, X):
        X = tf.matmul(X, self.W)
        X = X + self.b if self.bias else X
        X = self.activation(X) if self.activation is not None else X
        return X


class GatedRNN(object):
    def __init__(self, D, M, K, folder=''):
        self.D = D
        self.M = M  # hidden layer size
        self.K = K
        self.build()
        # for saving (and loading) graph Variables
        self.path = getcwd() + '/' + folder + '/'
        self.saver = tf.train.Saver()

    def build(self):
        # layers and parameters
        self.rnnUnit = GRU(self.D, self.M)
        self.logistic = HiddenLayer(self.M, self.K, activation=None)
        self.layers = [self.rnnUnit, self.logistic]
        self.params = [p for layer in self.layers for p in layer.params]

        # graph data placeholders
        self.tfX = tf.placeholder(tf.float32, shape=(None, self.D), name='X')
        self.tfY = tf.placeholder(tf.int32, shape=(None,), name='Y')

        # graph functions
        # full forward pass to output (logits, cost will do the softmax)
        self.logits = self.logistic.forward(self.rnnUnit.scanner(self.tfX))
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.tfY[-1],  # only care about last prediction
                logits=self.logits[-1],  # whole sequence into account
            )
        )
        self.predict_op = tf.argmax(self.logits, axis=1)

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, lr=1e-2, epochs=40,
            test_every=1, show_fig=False, save_model=False, load_model=False):

        Ntrain, Ntest = len(Xtrain), len(Xtest)
        Ytrain = [np.broadcast_to(Ytrain[j], Xtrain[j].shape[0])
                  for j in range(Ytrain.shape[0])]
        Ytest = [np.broadcast_to(Ytest[j], Xtest[j].shape[0])
                 for j in range(Ytest.shape[0])]

        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            if load_model:
                self.saver.restore(sess, self.path+'model.ckpt')

            train_costs, train_accs, test_costs, test_accs = [], [], [], []
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for s, (Xset, Yset, N) in enumerate([(Xtrain, Ytrain, Ntrain),
                                                    (Xtest, Ytest, Ntest)]):
                    n_correct = 0
                    epoch_cost = 0
                    # option to not validate on every epoch
                    if s and i % test_every != 0:
                        continue

                    # run through one sample at a time
                    for j in range(N):
                        if not s:
                            _, c, p = sess.run(
                                [self.train_op, self.cost, self.predict_op],
                                feed_dict={self.tfX: Xset[j],
                                           self.tfY: Yset[j]})
                        else:
                            c, p = sess.run(
                                [self.cost, self.predict_op],
                                feed_dict={self.tfX: Xset[j],
                                           self.tfY: Yset[j]})

                        epoch_cost += c

                        # log if final prediction was correct
                        if p[-1] == Yset[j][-1]:
                            n_correct += 1

                    if not s:
                        train_costs.append(epoch_cost)
                        train_accs.append(n_correct/N)
                        print("training.... ", end='')
                    else:
                        test_costs.append(epoch_cost)
                        test_accs.append(n_correct/N)
                        print("validating.. ", end='')

                    print("epoch:", i, "cost:", epoch_cost,
                          "classification rate:", n_correct/N)

            if save_model:
                save_path = self.saver.save(sess, self.path+'model.ckpt')
                print("Model saved in path: %s" % save_path)

        if show_fig:
            validX = np.arange(0, len(train_costs), test_every)
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(train_costs, label='training')
            axes[0].plot(validX, test_costs, label='validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Cost')
            axes[0].legend()
            axes[1].plot(train_accs, label='training')
            axes[1].plot(validX, test_accs, label='validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            fig.tight_layout()
            plt.show()


def one_hot_sequences(sequences, V):
    hot_mats = []
    for seq in sequences:
        matrix = np.zeros((len(seq), V))
        matrix[np.arange(len(seq)), seq] = 1
        hot_mats.append(matrix)
    return hot_mats


def trainTestSplit(X, T, ratio=.5):
    """
    Shuffle dataset and split into training and validation sets given a
    train:test ratio.
    """
    X, T = shuffle(X, T)
    N = len(X)
    Xtrain, Ttrain = X[:int(N*ratio)], T[:int(N*ratio)]
    Xtest, Ttest = X[int(N*ratio):], T[int(N*ratio):]
    return Xtrain, Ttrain, Xtest, Ttest


def main():
    X, Y, current_idx = get_poetry_classifier_data(1000)

    V = 0
    for seq in X:
        V = np.max(seq) if np.max(seq) > V else V
    V += 1
    print('Number of unique tags:', V)

    X = one_hot_sequences(X, V)

    Xtrain, Ytrain, Xtest, Ytest = trainTestSplit(X, Y, ratio=.8)
    K = 2
    nodes = 50

    rnn = GatedRNN(V, nodes, K, folder='classifier_gru')
    rnn.fit(Xtrain, Ytrain, Xtest, Ytest, lr=1e-3, epochs=30, show_fig=True,
            test_every=2, save_model=False, load_model=False)


if __name__ == '__main__':
    main()
