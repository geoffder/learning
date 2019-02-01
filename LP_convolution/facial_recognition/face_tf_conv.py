import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle


class ConvPoolLayer(object):
    '''
    Generate a new convolution and maxpool layer.
    '''
    def __init__(self, filtW, filtH, fmapsIn, fmapsOut, pool_sz=2):
        'Initialize filters'
        self.shape = (filtW, filtH, fmapsIn, fmapsOut)
        self.pool_sz = pool_sz
        '''
        NOTE: the * in *self.shape unpacks the tuple, since np.random.randn
        takes dimensions as seperate integer arguments, not a tuple like zeros.
        '''
        W = np.random.randn(
                    *self.shape) * np.sqrt(2.0 / np.prod(self.shape[:-1]))
        self.W = tf.Variable(W.astype(np.float32))
        b = np.zeros(fmapsOut, dtype=np.float32)
        self.b = tf.Variable(b)
        self.params = [self.W, self.b]

    def forward(self, X):
        'Apply convolutions and max pooling'
        conv_out = tf.nn.conv2d(X, self.W, [1, 1, 1, 1], 'SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        p = self.pool_sz
        pool_out = tf.nn.max_pool(conv_out, [1, p, p, 1], [1, p, p, 1], 'SAME')
        return pool_out


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


class CNN(object):
    '''
    Generate convolution new neural network, convolution layers as shapes,
    and sizes of fully connected dense layers with corresponding pkeeps for
    dropout regularization of each layer.
    '''
    def __init__(self, conv_layer_shapes, hidden_layer_sizes, p_keep):
        self.conv_layer_shapes = conv_layer_shapes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep

    def fit(self, X, T, Xvalid, Tvalid, lr=1e-4, reg=1e-3, mu=0.99,
            decay=0.99999, epochs=40, batch_sz=200, print_every=50):
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

        N = X.shape[0]
        # initialize conv layers
        self.conv_layers = []
        for shape in self.conv_layer_shapes:
            c = ConvPoolLayer(shape[0], shape[1], shape[2], shape[3])
            self.conv_layers.append(c)

        # initialize hidden layers
        # calculate input features to dense layers from last conv layer
        width, height = X.shape[1], X.shape[2]
        _, _, _, num_fmaps = self.conv_layer_shapes[-1]
        pool_redux = np.prod([c.pool_sz for c in self.conv_layers])
        D = width//pool_redux * height//pool_redux * num_fmaps
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
        for c in self.conv_layers:
            self.params += c.params
        for h in self.hidden_layers:
            self.params += h.params

        # set up theano functions and variables
        inputs = tf.placeholder(tf.float32, shape=(None, 48, 48, 1),
                                name='inputs')
        labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        logits = self.forward(inputs, True)

        # softmax done within the cost function, not at the end of forward
        # rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )  # + rcost  # add regularization

        train_op = tf.train.RMSPropOptimizer(lr, decay=decay,
                                             momentum=mu).minimize(cost)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)

        '''
        Setting up the last tensor equation placeholders to build the graphs
        that will be used for computation. No values, training loop is next!
        '''
        prediction = self.predict_dropout(inputs)  # returns labels

        # validation cost will be calculated separately,
        # since nothing will be dropped
        test_logits = self.forward_dropout(inputs, False)
        test_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=test_logits,
                labels=labels
            )
        )  # + rcost

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
        for c in self.conv_layers:
            Z = c.forward(Z)
        Z = self.flatten(Z)
        for h in self.hidden_layers:
            Z = h.forward(Z, is_training)
        return tf.matmul(Z, self.W)

    @staticmethod
    def flatten(Z):
        shape = Z.get_shape().as_list()
        batch = tf.shape(Z)[0]  # gives the variable batch length
        return tf.reshape(Z, [batch, np.prod(shape[1:])])

    def predict(self, X):
        pY = self.forward(X, False)
        return tf.argmax(pY, 1)

    def forward_dropout(self, X, is_training):
        Z = X
        # convpool layers
        for c in self.conv_layers:
            Z = c.forward(Z)
        Z = self.flatten(Z)
        # dense layers
        if is_training:
            Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
            Z = h.forward(Z, is_training)
            if is_training:
                Z = tf.nn.dropout(Z, p)
        return tf.matmul(Z, self.W)  # logits of final layer.

    def predict_dropout(self, X):
        pY = self.forward_dropout(X, False)
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

    faces = np.zeros((pixels.shape[0], 48, 48, 1))  # one channel, B&W img
    for i, img in enumerate(pixels):
        faces[i, :, :, 0] = img.reshape(48, 48) / 255
    X, T = classRebalance(faces, df['emotion'].values[:maxN])
    print('X shape:', X.shape, 'T shape:', T.shape)
    print('emotion counts:', [(T == k).sum() for k in np.unique(T)])
    del df, pixels  # clean-up

    Xtrain, Ttrain, Xtest, Ttest = trainTestSplit(X, T, ratio=.8)
    ann = CNN([[5, 5, 1, 20], [5, 5, 20, 50], [5, 5, 50, 50]],
              [1000, 500, 300, 100],
              [0.8, 0.5, 0.5, .5, .5])

    ann.fit(Xtrain, Ttrain, Xtest, Ttest, lr=1e-3, epochs=100)


if __name__ == '__main__':
    main()
