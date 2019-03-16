import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import nltk
import string
import json
from os import getcwd

from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


"""
Try using the same scheme LP uses in the wikipedia data to trim down the vocab.
Make an UNKNOWN token that replaces all words that are not one of the common
words that made the cut-off.
"""


def init_weight(M1, M2):
    '''
    The weights being small is really important, my pytorch implementation did
    not work until I remembered to divide the randn weights like so.
    '''
    return np.random.randn(M1, M2) / np.sqrt(M1 + M2)


class LSTM(object):
    "Class Definition of Long Short-Term Memory Layer in Tensorflow"
    def __init__(self, M1, M2):
        self.M1 = M1  # input size
        self.M2 = M2  # hidden layer size
        self.build()

    def build(self):
        # input gate weights (and bias)
        self.Wxi = tf.Variable(
            init_weight(self.M1, self.M2).astype(np.float32))
        self.Whi = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.Wci = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.bi = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # forget gate weights (and bias)
        self.Wxf = tf.Variable(
            init_weight(self.M1, self.M2).astype(np.float32))
        self.Whf = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.Wcf = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.bf = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # memory cell weights (and bias)
        self.Wxc = tf.Variable(
            init_weight(self.M1, self.M2).astype(np.float32))
        self.Whc = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.bc = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # output gate weights (and bias)
        self.Wxo = tf.Variable(
            init_weight(self.M1, self.M2).astype(np.float32))
        self.Who = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.Wco = tf.Variable(
            init_weight(self.M2, self.M2).astype(np.float32))
        self.bo = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # initial memory cell and hidden repesentation
        self.c0 = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        self.h0 = tf.Variable(np.zeros(self.M2, dtype=np.float32))
        # param collection
        self.params = [
            self.Wxi, self.Whi, self.Wci, self.bi,
            self.Wxf, self.Whf, self.Wcf, self.bf,
            self.Wxc, self.Whc, self.bc,
            self.Wxo, self.Who, self.Wco, self.bo,
            self.h0
        ]

    def recurrence(self, last, new):
        # reshape recurrent inputs
        last_h, last_c = last
        last_h = tf.reshape(last_h, (1, self.M2))
        last_c = tf.reshape(last_c, (1, self.M2))
        new = tf.reshape(new, (1, -1))
        # calculate input and forget gates
        input = tf.sigmoid(
            tf.matmul(new, self.Wxi) + tf.matmul(last_h, self.Whi)
            + tf.matmul(last_c, self.Wci) + self.bi)
        forget = tf.sigmoid(
            tf.matmul(new, self.Wxf) + tf.matmul(last_h, self.Whf)
            + tf.matmul(last_c, self.Wcf) + self.bf)
        # calculate new memory cell value using input and forget gates
        c_hat = tf.tanh(tf.matmul(new, self.Wxc) + tf.matmul(last_h, self.Whc)
                        + self.bc)
        cell = forget*last_c + input*c_hat
        # calculate output gate
        output = tf.sigmoid(
            tf.matmul(new, self.Wxo) + tf.matmul(last_h, self.Who)
            + tf.matmul(cell, self.Wco) + self.bo)
        # update hidden representation
        hidden = output*tf.tanh(cell)
        return (tf.reshape(hidden, (self.M2,)), tf.reshape(cell, (self.M2,)))

    def scanner(self, X):
        """
        The recurrent loop of this RNN layer. Tuple of h0 and c0, as we
        will be returning both a hidden representation value [h(t)] and
        a memory cell value [c(t)] with each loop. We don't want the memory
        cell outside of the scope of this layer, so we return only the
        first element of the output tuple, h0.
        """
        self.scan = tf.scan(
            fn=self.recurrence,  # run this on each element of the input
            elems=X,
            initializer=(self.h0, self.c0),  # zeros
        )
        return self.scan[0]  # only want hidden for output (ignore cell)


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


class Embedder(object):
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
        # self.rnnUnit = GRU(self.D, self.M)
        self.rnnUnit = LSTM(self.D, self.M)
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
            save_model=False, load_model=False, save_embeddings=False):

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
            if save_embeddings:
                embeddings = sess.run(self.embed.W)
                np.save(self.path+'word_embeddings', embeddings)

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


def my_tokenizer(s):
    '''
    Break input text in to a list of words, lemmatize them, and remove those
    that will not be useful. Short words, stopwords, etc.
    '''
    stopwords = {
        "ccb", "n't", "'ve", "'re", "pause", "wa", "'s"
    }
    s = s.lower()  # downcase
    s.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.tokenize.word_tokenize(s)  # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in stopwords]  # remove stopwords
    # tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


# maybe add something in here to remove lines by CCB
def process(lines):
    word_index_map = {'START': 0, 'END': 1}
    current_index = 2
    all_tokens = []
    all_lines = []
    index_word_map = []
    error_count = 0
    for line in lines:
        try:
            # this will throw exception if bad characters
            line = line.encode('ascii', 'ignore').decode('utf-8')
            all_lines.append(line)
            tokens = my_tokenizer(line)
            all_tokens.append(tokens)
            for token in tokens:
                if token not in word_index_map:
                    word_index_map[token] = current_index
                    current_index += 1
                    index_word_map.append(token)
        except Exception as e:
            print(e)
            print(line)
            error_count += 1

    return all_lines, all_tokens, word_index_map, index_word_map


def display_embeddings(folder=''):
    # load saved embeddings and word2idx dict
    embeddings = np.load(folder+'word_embeddings.npy')
    with open(folder+'word2idx.json', 'r') as fp:
        word2idx = json.load(fp)
    idx2word = {idx: word for word, idx in word2idx.items()}

    # embed with t-SNE into 2 dimensions
    reduced = TSNE(n_components=2).fit_transform(embeddings)

    # reduce dimensionality with PCA
    # pca = PCA()
    # reduced = pca.fit_transform(embeddings)
    # cumulative = np.cumsum(pca.explained_variance_ratio_)
    # plt.plot(cumulative)
    # plt.title('Cumulative Information')
    # plt.xlabel('dimensions')
    # plt.ylabel('variance explained')

    # plot the dimensional reduction of the data
    fig, ax = plt.subplots(1)
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=.5, s=10, c='black')
    for i in range(reduced.shape[0]):
        ax.annotate(s=idx2word[i], xy=(reduced[i, 0], reduced[i, 1]))
    ax.set_title('2D Word Embeddings')
    ax.set_xlabel('dimension 1')
    ax.set_ylabel('dimension 2')
    plt.show()


def train_embeddings():
    path = '..\\large_files\\Transcripts\\'
    interviews = {
        'chiros': ['CCP1TRUE', 'CCP2', 'CCP3'],
        'chiroClients': ['CC1'],
        'instructors': ['INSTRUCTOR1', 'INSTRUCTOR2'],
        'yogis': ['SKJ%s' % (i) for i in range(1, 10)],
    }

    # process text into arrays of word tokens, and idx dicts
    all_raw = []
    for subject in interviews['yogis']:
        all_raw += [line.rstrip() for line in open(path + subject + '.txt')]
    all_lines, all_tokens, word2idx, idx2word = process(all_raw)
    all_tokens = [ele for ele in all_tokens if len(ele) > 3]

    # word sequence vectors
    sequences = np.array([[word2idx[w] for w in sentence]
                         for sentence in all_tokens])
    N = len(sequences)  # number of lines of tokens
    V = len(word2idx)  # len returns number of pairs in dict
    print('Number of Sentences:', N, 'Vocabulary size:', V)
    D = 100  # 50  # embedding dimensions
    nodes = 100  # 50  # hidden nodes

    rnn = Embedder(V, D, nodes, folder='GRU_yogis')
    rnn.fit(sequences, word2idx, lr=1e-3, epochs=5, show_fig=False,
            save_model=False, load_model=True, save_embeddings=True)

    with open(rnn.path+'word2idx.json', 'w') as fp:
        json.dump(word2idx, fp)


if __name__ == '__main__':
    # train_embeddings()
    display_embeddings(folder='GRU_yogis/')
