import numpy as np
import tensorflow as tf


def adder(a, b):
    'Cumulatively add elements of array. a is sum so far, b is next element.'
    return np.add(a, b)


def squarer(a, b):
    'Square each element of the array, do not use last output.'
    return np.multiply(b, b)


def fibonacci(a, b):
    'Calculate Fibocacci sequence. Sum previous two values, store new value.'
    return tf.stack([b, tf.math.reduce_sum(a)], axis=0)


x = np.arange(10)
xtensor = tf.Variable(np.arange(15))
fibo_init = tf.Variable([1, 1])  # [t-2, t-1], t-1 is last sum.

# cumulative sum
sum = tf.scan(adder, x)
# each element squared
square = tf.scan(squarer, x)
# fibonacci sequence
fibo = tf.scan(fibonacci, xtensor, initializer=fibo_init)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('cumsum:', sess.run(sum))
    print('squared:', sess.run(square))
    print('fibonacci:', sess.run(fibo)[:, 1])
