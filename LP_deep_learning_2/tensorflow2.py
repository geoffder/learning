import tensorflow as tf
import matplotlib.pyplot as plt
from LP_util import get_normalized_data, y2indicator


def main():
    Xtrain, Xtest, Ttrain_label, Ttest_label = get_normalized_data()
    Ttrain, Ttest = y2indicator(Ttrain_label), y2indicator(Ttest_label)

    lr = .0004
    # reg = .01

    max_iter = 20
    print_period = 10

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M1 = 300  # hidden units
    M2 = 100
    K = 10  # output classes

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.1))

    def forward(X, W1, b1, W2, b2, W3, b3):
        Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
        return tf.matmul(Z2, W3) + b3  # return activation, not softmax

    def error_rate(P, T):
        return (P != T).mean()

    tfX = tf.placeholder(tf.float32, [None, D])
    tfT = tf.placeholder(tf.float32, [None, K])

    W1 = init_weights([D, M1])
    b1 = init_weights([M1])
    W2 = init_weights([M1, M2])
    b2 = init_weights([M2])
    W3 = init_weights([M2, K])
    b3 = init_weights([K])

    tfY = forward(tfX, W1, b1, W2, b2, W3, b3)
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tfT, logits=tfY))
    predict_op = tf.argmax(tfY, axis=1)
    # train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    train_op = tf.train.RMSPropOptimizer(
                lr, decay=.99, momentum=.9).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    LL = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
            Tbatch = Ttrain[j*batch_sz:(j*batch_sz+batch_sz)]

            sess.run(train_op, feed_dict={tfX: Xbatch, tfT: Tbatch})
            if j % print_period == 0:
                Ptest = sess.run(predict_op, feed_dict={tfX: Xtest})
                err = error_rate(Ptest, Ttest_label)
                c = sess.run(cost, feed_dict={tfX: Xtest, tfT: Ttest})
                print("cost / err at iteration i=%d, j=%d: %.3f / %.3f"
                      % (i, j, c, err))
                LL.append(c)

    Ptest = sess.run(predict_op, feed_dict={tfX: Xtest})
    print("final error rate:", error_rate(Ptest, Ttest_label))
    plt.plot(LL)
    plt.show()


if __name__ == "__main__":
    main()
