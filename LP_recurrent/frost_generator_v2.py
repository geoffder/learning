import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from LP_util import get_robert_frost
from os import getcwd

'''
Language model that learns from Robert Frost poems (robert_frost.txt) to create
word embeddings that can be used to generate Robert Frost flavoured gibberish.
Similar architecture to the SimpleRNN used for the parity problem, but with an
additional embedding layer at the bottom (beginning). Sentences are converted
from integer arrays (indices) to one-hot vocab matrices, then fed through the
network. Trained in an unsupervised manner, trying to predict the next word,
given the previous words. This v2 is simply a cleaned up version of v1, with
more graph building moved in to build(), rather than living in fit(). Also,
using the same number of RNN nodes as embedding dimensions, leading to much
higher classification rate. Possibly overfit? But v1 with M=20 was p silly.
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
    def __init__(self, V, D, M, folder=''):
        self.V = V
        self.D = D
        self.M = M  # hidden layer size
        self.build()
        # for saving (and loading) graph Variables
        self.path = getcwd() + '/' + folder + '/'
        self.saver = tf.train.Saver()

    def build(self):
        # layers and parameters
        self.embed = HiddenLayer(self.V, self.D, bias=False, activation=None)
        self.rnnUnit = RNNunit(self.D, self.M)
        self.logistic = HiddenLayer(self.M, self.V, activation=None)
        self.layers = [self.embed, self.rnnUnit, self.logistic]
        self.params = [p for layer in self.layers for p in layer.params]

        # graph data placeholders
        self.tfX = tf.placeholder(tf.float32, shape=(None, self.V), name='X')
        self.tfY = tf.placeholder(tf.int32, shape=(None,), name='Y')

        # graph functions
        # full forward pass to output (logits, cost will do the softmax)
        self.logits = self.logistic.forward(
            self.rnnUnit.scanner(self.embed.forward(self.tfX)))
        # softmax of logits for poem-generation probabiliities
        self.output_probs = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.tfY,
                logits=self.logits,
            )
        )
        self.predict_op = tf.argmax(self.logits, axis=1)

    def fit(self, sentences, word2idx, lr=1e-2, epochs=100, show_fig=False,
            save_model=False, load_model=False):

        N = len(sentences)
        X, Y = self.vector_matrices(sentences, word2idx)

        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            if load_model:
                self.saver.restore(sess, self.path+'model.ckpt')

            epoch_costs, epoch_accs = [], []
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                accuracy = 0
                epoch_cost = 0
                # run through one sample at a time
                for j in range(N):
                    _, c, p = sess.run(
                        [self.train_op, self.cost, self.predict_op],
                        feed_dict={self.tfX: X[j], self.tfY: Y[j]})

                    epoch_cost += c

                    # calculate % accuracy for this epoch
                    accuracy += np.sum(p == Y[j]) / p.shape

                epoch_costs.append(epoch_cost)
                epoch_accs.append(accuracy/N)
                print("epoch:", i, "cost:", epoch_cost,
                      "classification rate:", (accuracy/N))

            if save_model:
                save_path = self.saver.save(sess, self.path+'model.ckpt')
                print("Model saved in path: %s" % save_path)

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

    def generate(self, init_word_prob, word2idx, load_model=False):
        # convert word2idx -> idx2word
        idx2word = {v: k for k, v in word2idx.items()}
        V = len(init_word_prob)

        # generate 4 lines at a time
        n_lines = 0

        # pick a word to start the line based on frequency of each word
        # starting lines in the training dataset
        X = np.zeros((1, V))  # do one-hot because of how I built the net
        # can see the benefit of feeding the network the idxs better now
        word_idx = np.random.choice(V, p=init_word_prob)
        X[0, word_idx] = 1

        print(idx2word[word_idx], end=" ")  # end='\n' by default

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            if load_model:
                self.saver.restore(sess, self.path+'model.ckpt')
            while n_lines < 4:
                next_probs = sess.run(self.output_probs,
                                      feed_dict={self.tfX: X})[-1]  # last one
                # draw the next word based on predicted probabilities
                word_idx = np.random.choice(V, p=next_probs)
                one_hot = np.zeros((1, V))
                one_hot[0, word_idx] = 1
                X = np.concatenate([X, one_hot], axis=0)
                if word_idx > 1:
                    # it's a real word, not start/end token
                    word = idx2word[word_idx]
                    print(word, end=" ")
                elif word_idx == 1:
                    # end of line predicted (end token)
                    n_lines += 1
                    print('')  # moves to next line of output in terminal
                    if n_lines < 4:
                        # reset to start of line
                        X = np.zeros((1, V))
                        word_idx = np.random.choice(V, p=init_word_prob)
                        X[0, word_idx] = 1
                        print(idx2word[word_idx], end=" ")

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
    nodes = 50
    rnn = LanguageRNN(V, D, nodes, folder='frost_model_v2')
    rnn.fit(sentences, word2idx, lr=lr, epochs=epochs, show_fig=True,
            save_model=False, load_model=True)

    return rnn


def generate_poem(model=None):
    sentences, word2idx = get_robert_frost()
    V = len(word2idx)  # len returns number of pairs in dict
    D = 50  # embedding dimensions
    nodes = 50

    # determine initial state distribution for starting sentences
    # num of appearances of each word at start of line divided by num sentences
    init_word_prob = np.zeros(V)
    for sentence in sentences:
        init_word_prob[sentence[0]] += 1
    init_word_prob /= init_word_prob.sum()

    if not model:
        model = LanguageRNN(V, D, nodes, folder='frost_model_v2')
        model.generate(init_word_prob, word2idx, load_model=True)
    else:
        model.generate(init_word_prob, word2idx, load_model=False)


if __name__ == '__main__':
    # rnn = train_language(lr=1e-3, epochs=100)
    # generate_poem(rnn)
    generate_poem()
