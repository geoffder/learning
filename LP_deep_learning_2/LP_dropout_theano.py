import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
from util import get_normalized_data
from sklearn.utils import shuffle

'''
Notes added by myself as learning exercise and practice. Code by the
Lazy Programmer for Deep Learning Part 2 (Modern Deep Learning in Python).
'''


class HiddenLayer(object):
    '''
    Generate new hidden layer with input size and number of nodes as args,
    as well as a string to serve as an ID.
    '''
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1  # input layer size
        self.M2 = M2  # size of this new layer
        W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)  # weights
        b = np.zeros(M2)  # bias
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        '''Calculate values for next layer with ReLU activation'''
        return T.nnet.relu(X.dot(self.W) + self.b)


class ANN(object):
    '''
    Generate new neural network, taking hidden layer sizes as a list, and
    dropout rate as p_keep
    '''
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-4, mu=0.9, decay=0.9,
            epochs=8, batch_sz=100, show_fig=False):
        '''
        Takes training data and test data (valid) at once, then trains and
        validates along the way. Modifying hyperparams of learning_rate, mu,
        decay, epochs (iterations = N//batch_sz * epochs), batch_sz and whether
        to display a figure are passed as optional variables.
        '''
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int32)

        self.rng = RandomStreams()

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D  # first input layer is the number of features in X
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)  # layer ID is just the number
            self.hidden_layers.append(h)
            M1 = M2  # input layer to next layer is this layer.
            count += 1
        # output layer weights (last hidden layer to K output classes)
        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY_train = self.forward_train(thX)  # function to calc prob Y given X

        # this cost is for training
        cost = -T.mean(T.log(pY_train[T.arange(thY.shape[0]), thY]))

        # gradients wrt each param
        grads = T.grad(cost, self.params)

        # for momentum
        '''
        np.zeros_like(array) returns an array(/matrix) of the same shape and
        type of the given array. Very cool, never seen this before.
        '''
        dparams = [theano.shared(np.zeros_like(p.get_value()))
                   for p in self.params]

        # for rmsprop, initialize cache as 1
        cache = [theano.shared(np.ones_like(p.get_value()))
                 for p in self.params]
        '''
        Noting for myself that I've never seen this way of using zip to loop
        through multiple lists/arays with the same indices simultaneously.
        Makes a lot of sense now, I should see where I can use this to turn
        loops over indices in my code in to list comprehension that is by ele.
        '''
        # these are the functions for updating the variables of
        # dparams (momentum) and cache.
        new_cache = [decay*c + (1-decay)*g*g
                     for p, c, g in zip(self.params, cache, grads)]
        new_dparams = [mu*dp - learning_rate*g/T.sqrt(new_c + 1e-10)
                       for p, new_c, dp, g in zip(
                       self.params, new_cache, dparams, grads)]

        '''
        Using zip to create lists of tuples of the variables themselves, and
        the fuctions for updating them (cache, momentum params and params),
        where params are weights (W) and biases (b) for each layer.
        '''
        updates = [
            (c, new_c) for c, new_c in zip(cache, new_cache)
        ] + [
            (dp, new_dp) for dp, new_dp in zip(dparams, new_dparams)
        ] + [
            (p, p + new_dp) for p, new_dp in zip(self.params, new_dparams)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        # for evaluation and prediction, more theano graph set-up with tensors
        # still no values yet in any of these. Training loop next!
        pY_predict = self.forward_predict(thX)
        cost_predict = -T.mean(T.log(pY_predict[T.arange(thY.shape[0]), thY]))
        prediction = self.predict(thX)
        cost_predict_op = theano.function(inputs=[thX, thY],
                                          outputs=[cost_predict, prediction])

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                # theano function defined above that does all the work.
                # takes the data (like feed_dict in tf). The update calcs were
                # given to it above as a list for all layers.
                train_op(Xbatch, Ybatch)

                if j % 50 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c,
                          "error rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward_train(self, X):
        Z = X
        '''
        Here is the dropout implementation. A mask is generated for each hidden
        layer, then multiplied against the values of the nodes.

        The second index of the slice is EXCLUSIVE, so this is for all rates
        EXCEPT for the last one. The output layer is logistic regression,
        not the same forward function as the rest of the layers, so we do
        it outside of the loop.
        '''
        for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
            mask = self.rng.binomial(n=1, p=p, size=Z.shape)
            Z = mask * Z
            Z = h.forward(Z)  # forward function of the layer object
        mask = self.rng.binomial(n=1, p=self.dropout_rates[-1], size=Z.shape)
        Z = mask * Z
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def forward_predict(self, X):
        '''
        forward_predict is seperate than forward_train because of dropout.
        Here, instead of masking the layers, we multiply the value of the nodes
        by p_keep (1 - p_drop).
        '''
        Z = X
        for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
            Z = h.forward(p * Z)
        return T.nnet.softmax((self.dropout_rates[-1] * Z).dot(self.W)
                              + self.b)

    def predict(self, X):
        pY = self.forward_predict(X)
        return T.argmax(pY, axis=1)


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    # hidden layer sizes and dropout rates (keep rate)
    ann = ANN([500, 300], [0.8, 0.5, 0.5])
    ann.fit(Xtrain, Ytrain, Xtest, Ytest, show_fig=True)


if __name__ == '__main__':
    main()