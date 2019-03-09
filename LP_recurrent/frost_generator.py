import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from LP_util import get_robert_frost

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


class HiddenLayer(object):
    '''
    Simple layer with optional non-linearity, and bias. For example, can be
    used to get logits by setting activation=None.
    '''
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


class LanguageRNN(object):
    def __init__(self, V, D, M):
        self.V = V
        self.D = D
        self.M = M  # hidden layer size
        self.build()

    def build(self):
        self.embed = HiddenLayer(self.V, self.D, bias=False, activation=None)
        self.rnnUnit = RNNunit(self.D, self.M)
        self.logistic = HiddenLayer(self.M, self.V, activation=None)
        self.layers = [self.embed, self.rnnUnit, self.logistic]
        self.params = [p for layer in self.layers for p in layer.params]

    def fit(self, sentences, word2idx, lr=1e-2, epochs=100, show_fig=False):
        N, V = len(sentences), len(word2idx)

        X, Y = self.vector_matrices(sentences, word2idx)

        tfX = tf.placeholder(tf.float32, shape=(None, V), name='X')
        tfY = tf.placeholder(tf.int32, shape=(None,), name='Y')

        # full forward pass to output (logits, cost will do the softmax)
        logits = self.logistic.forward(
            self.rnnUnit.scanner(self.embed.forward(tfX)))

        # LP used the method below as cost, only calculating probabilities
        # for a sample of the vocabulary at a time.
        # nce_weights = tf.transpose(
        #     self.logistic.W, [1, 0])  # needs to be VxD, not DxV
        # nce_biases = self.logistic.b
        #
        # h = tf.reshape(self.rnnUnit.scanner(
        #     self.embed.forward(tfX)), (-1, self.M))
        # labels = tf.reshape(tfY, (-1, 1))
        #
        # cost = tf.reduce_mean(
        #     tf.nn.sampled_softmax_loss(
        #         weights=nce_weights,
        #         biases=nce_biases,
        #         labels=labels,
        #         inputs=h,
        #         num_sampled=50,  # number of negative samples
        #         num_classes=V
        #     )
        # )

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

            epoch_costs, epoch_accs = [], []
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                accuracy = 0
                epoch_cost = 0
                # run through one sample at a time
                for j in range(N):
                    _, c, p = sess.run(
                        [train_op, cost, predict_op],
                        feed_dict={tfX: X[j], tfY: Y[j]})
                    # _, c = sess.run(
                    #     [train_op, cost],
                    #     feed_dict={tfX: X[j], tfY: Y[j]})

                    epoch_cost += c

                    # calculate % accuracy for this epoch
                    accuracy += np.sum(p == Y[j]) / p.shape

                epoch_costs.append(epoch_cost)
                epoch_accs.append(accuracy/N)
                print("epoch:", i, "cost:", epoch_cost,
                      "classification rate:", (accuracy/N))
                # for j in range(10):
                #     p = sess.run(
                #         predict_op, feed_dict={tfX: X[j], tfY: Y[j]})
                #     accuracy += np.sum(p == Y[j]) / p.shape
                # print("epoch:", i, "cost:", epoch_cost,
                #       "classification rate:", (accuracy/10))

        if show_fig:
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(epoch_costs)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Cost')
            axes[1].plot(epoch_accs)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            fig.tight_layout()
            plt.show()

    @staticmethod
    def one_hot_sentences(sentences, V):
        hot_mats = []
        for sentence in sentences:
            matrix = np.zeros((len(sentence), V))
            matrix[np.arange(len(sentence)), sentence] = 1
            hot_mats.append(matrix)
        return hot_mats

    def vector_matrices(self, sentences, word2idx):
        V = len(word2idx)
        hot_mats = self.one_hot_sentences(sentences, V)
        startVec = np.zeros(V).reshape(1, V)
        startVec[0, word2idx['START']] = 1
        X = [np.concatenate([startVec, sample], axis=0) for sample in hot_mats]
        Y = [np.concatenate([s, [word2idx['END']]]) for s in sentences]
        return X, Y


def train_language(lr=1e-2, epochs=200):
    sentences, word2idx = get_robert_frost()

    N = len(sentences)  # number of lines of poetry
    V = len(word2idx)  # len returns number of pairs in dict
    print('Number of Sentences:', N, 'Vocabulary size:', V)
    D = 50  # embedding dimensions
    nodes = 20
    rnn = LanguageRNN(V, D, nodes)
    rnn.fit(sentences, word2idx, lr=lr, epochs=epochs, show_fig=True)

    return rnn


def generate_poems(rnn):
    pass


if __name__ == '__main__':
    rnn = train_language(lr=1e-3)
    generate_poems(rnn)
