import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from LP_util import all_parity_pairs_with_sequence_labels

'''
Parity problem with a simple recurrent neural net. Based off of the tensorflow
implementation by Lazy Programmer, but restructured into classes with better
reusability and additional notes.
'''


def init_weight(M1, M2):
    '''
    The weights being small is really important, my pytorch implementation did
    not work until I remembered to divide the randn weights like so.
    '''
    return np.random.randn(M1, M2) / np.sqrt(M1 + M2)


class RNNunit(object):
        def __init__(self, M1, M2):
            self.M1 = M1  # input size
            self.M2 = M2  # hidden layer size
            self.build()

        def build(self):
            # input weight
            self.Wx = tf.Variable(
                init_weight(self.M1, self.M2).astype(np.float32))
            # hidden weight
            self.Wh = tf.Variable(
                init_weight(self.M2, self.M2).astype(np.float32))
            # hidden bias
            self.bh = tf.Variable(np.zeros(self.M2, dtype=np.float32))
            # initial hidden repesentation
            self.h0 = tf.Variable(np.zeros(self.M2, dtype=np.float32))
            self.params = [self.Wx, self.Wh, self.bh, self.h0]

        def forward(self, X):
            'Multiply X with input weights.'
            return tf.matmul(X, self.Wx)

        def recurrence(self, last, new):
            # reshape recurrent input, since output is shape (M2,), and new
            # has the shape of (1, M2), since it is a single timestep with M2
            # dimensions (X already multiplied with Wx)
            last = tf.reshape(last, (1, self.M2))
            hidden = tf.nn.relu(new + tf.matmul(last, self.Wh) + self.bh)
            return tf.reshape(hidden, (self.M2,))

        def scanner(self, X):
            '''
            The recurrent loop of this simple RNN layer. h0 is the "last" arg
            for the first element of the input. We are initializing at zero.
            '''
            self.scan = tf.scan(
                fn=self.recurrence,  # run this on each element of the input
                elems=self.forward(X),  # X @ Wx (input weights)
                initializer=self.h0,  # zeros
            )
            return self.scan


class LogisticLayer(object):
    'Simple layer without a non-linearity, use for getting output logits.'
    def __init__(self, M1, M2):
        self.M1 = M1  # input size
        self.M2 = M2  # hidden layer size
        self.build()

    def build(self):
        self.W = tf.Variable(init_weight(self.M1, self.M2).astype(np.float32))
        self.b = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.matmul(X, self.W) + self.b


class SimpleRNN:
    def __init__(self, D, M, K):
        self.D = D
        self.M = M  # hidden layer size
        self.K = K
        self.build()

    def build(self):
        self.rnnUnit = RNNunit(self.D, self.M)
        self.logistic = LogisticLayer(self.M, self.K)

    def fit(self, X, Y, lr=1e-2, epochs=100, show_fig=False):
        N, T, D = X.shape

        # X and Y will be fed in to the graph one sample at a time, thus shape
        # is (T, D)/(Time, Dimension) for X and (T,) for Y (labels each step)
        tfX = tf.placeholder(tf.float32, shape=(T, D), name='X')
        tfY = tf.placeholder(tf.int32, shape=(T,), name='Y')

        # forward pass to output (logits, cost will do the softmax)
        logits = self.logistic.forward(self.rnnUnit.scanner(tfX))

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tfY,
                logits=logits,
            )
        )

        predict_op = tf.argmax(logits, axis=1)
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            sample_costs = []
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                n_correct = 0
                epoch_cost = 0
                # run through one sample at a time
                for j in range(N):
                    # reshape single sample sequence to (Time, Dimension)
                    _, c, p = sess.run(
                        [train_op, cost, predict_op],
                        feed_dict={tfX: X[j].reshape(T, D), tfY: Y[j]})
                    epoch_cost += c
                    sample_costs.append(c)
                    # check if final prediction is correct for current sample
                    if p[-1] == Y[j, -1]:
                        n_correct += 1

                print("i:", i, "cost:", epoch_cost,
                      "classification rate:", (float(n_correct)/N))

                # stop if perfect accuracy is attained
                if n_correct == N:
                    break

        if show_fig:
            plt.plot(sample_costs)
            plt.xlabel('Samples Ran')
            plt.ylabel('Sample Cost')
            plt.show()


def parity(B=12, lr=1e-2, epochs=200):
    X, Y = all_parity_pairs_with_sequence_labels(B)
    X = X.astype(np.float32)

    N, T, D = X.shape
    print('X shape:', X.shape, 'Y shape:', Y.shape)
    nodes = 20
    K = 2
    rnn = SimpleRNN(D, nodes, K)
    rnn.fit(X, Y, lr=lr, epochs=epochs, show_fig=True)


if __name__ == '__main__':
    parity()
