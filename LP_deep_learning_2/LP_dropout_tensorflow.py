import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from LP_util import get_normalized_data
from sklearn.utils import shuffle

'''
Notes added by myself as learning exercise and practice. Code by the
Lazy Programmer for Deep Learning Part 2 (Modern Deep Learning in Python).
'''


class HiddenLayer(object):
    '''
    Generate new hidden layer with input size and number of nodes as args.
    '''
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2
        W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        '''Calculate values for next layer with ReLU activation'''
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    '''
    Generate new neural network, taking hidden layer sizes as a list, and
    dropout rate as p_keep
    '''
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep

    def fit(self, X, Y, Xvalid, Yvalid, lr=1e-4, mu=0.9, decay=0.9, epochs=15,
            batch_sz=100, print_every=50):
        '''
        Takes training data and test data (valid) at once, then trains and
        validates along the way. Modifying hyperparams of learning_rate, mu,
        decay, epochs (iterations = N//batch_sz * epochs), batch_sz and how
        often to validate and print results are passed as optional variables.
        '''
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int64)

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
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
        logits = self.forward(inputs)  # logits then fed in to the cost func

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
        test_logits = self.forward_test(inputs)
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
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                sess.run(train_op, feed_dict={inputs: Xbatch,
                                              labels: Ybatch})

                if j % print_every == 0:
                    c = sess.run(test_cost, feed_dict={inputs: Xvalid,
                                                       labels: Yvalid})
                    p = sess.run(prediction, feed_dict={inputs: Xvalid})
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c,
                          "error rate:", e)

        plt.plot(costs)
        plt.show()

    def forward(self, X):
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
            Z = h.forward(Z)
            Z = tf.nn.dropout(Z, p)
        return tf.matmul(Z, self.W) + self.b  # logits of final layer.

    def forward_test(self, X):
        '''
        Note that inputs aren't scaled here, the dropout function already
        took care of that during training (see LP's note above).
        '''
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward_test(X)
        return tf.argmax(pY, 1)


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    ann = ANN([500, 300], [0.8, 0.5, 0.5])
    ann.fit(Xtrain, Ytrain, Xtest, Ytest)


if __name__ == '__main__':
    main()
