import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Nclass = 500
D = 2 # number of input features / dimensionality
M = 3 # hidden layer size
K = 3 # number of classes

# three gaussian clouds
X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Tlabel = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
N = len(Tlabel)

T = np.zeros((N, K))
T[np.arange(N), Tlabel.astype(np.int32)] = 1

# tensorflow has it's own kind of variables, so define weights using tf
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2 # return activation, not softmax
    # when we do the costs later, we want the 'logits' as inputs not the output of the softmax

# tensorflow placeholders
# creates graphs so it knows how to work with the data, before doing calculations
tfX = tf.placeholder(tf.float32, [None, D]) # can pass in any size N, but D is set
tfT = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D,M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

Y = forward(tfX, W1, b1, W2, b2) # tfX doesn't have a value yet

# define the cost function some tf built in functions
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=T, logits=Y))

lrn_rate = .05
# gradient descent is solved automatically, given the defined cost function above
train_op = tf.train.GradientDescentOptimizer(lrn_rate).minimize(cost)
predict_op = tf.argmax(Y, axis=1) #axis=1

# set up a model session
sess = tf.Session() # not explained yet, but we need it. (object that everything is run in)
init = tf.global_variables_initializer() # initializes previously defined variables (weights)
sess.run(init)

for i in range(1000):
    # input is a dict with keys=placeholders, values=the values to feed to the placeholders
    sess.run(train_op, feed_dict={tfX: X, tfT: T}) # run a round of gradient descent
    # it calculates Y again and uses it as the input automatically I guess
    pred = sess.run(predict_op, feed_dict={tfX: X, tfT: T}) # calculate predicted values with current weights
    if i % 10 == 0:
        #print((pred == np.argmax(T, axis=1)).mean())
        print((pred == Tlabel).mean())
