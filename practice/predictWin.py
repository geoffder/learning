import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from get_prices import getPrices

# load market history data and winner labels
#X, T = getPrices()
# alternatively load data from file
X, T = np.load('priceData.npy'), np.load('winners.npy')
# print(X.shape, Y.shape) # X(931, 1000, 18) T(931, 3) one-hot indicator matrix

# get class label version of T
Tlabels = np.argmax(T, axis=1)

# deal with class imbalances
# break up dataset by class
X_btc = X[Tlabels == 0]
T_btc = T[Tlabels == 0]
X_eth = X[Tlabels == 1]
T_eth = T[Tlabels == 1]
X_ltc = X[Tlabels == 2]
T_ltc = T[Tlabels == 2]

# calculate target N (largest class)
bigN = np.max([X_btc.shape[0],X_eth.shape[0],X_ltc.shape[0]])

# roughly equalize size of each class
X_btc = np.concatenate(([X_btc]*int(np.round(bigN/X_btc.shape[0]))), axis=0)
T_btc = np.concatenate(([T_btc]*int(np.round(bigN/T_btc.shape[0]))), axis=0)
X_eth = np.concatenate(([X_eth]*int(np.round(bigN/X_eth.shape[0]))), axis=0)
T_eth = np.concatenate(([T_eth]*int(np.round(bigN/T_eth.shape[0]))), axis=0)
X_ltc = np.concatenate(([X_ltc]*int(np.round(bigN/X_ltc.shape[0]))), axis=0)
T_ltc = np.concatenate(([T_ltc]*int(np.round(bigN/T_ltc.shape[0]))), axis=0)
#print(X_btc.shape[0],X_eth.shape[0],X_ltc.shape[0])

# recombine dataset
X = np.concatenate([X_btc, X_eth, X_ltc], axis=0)
T = np.concatenate([T_btc, T_eth, T_ltc], axis=0)
print(X.shape,T.shape)

# train and test sets
X, T = shuffle(X, T)
N_train = int(X.shape[0] * .8) #.8
X_train = X[:N_train-1]
T_train = T[:N_train-1]
X_test = X[N_train:]
T_test = T[N_train:]

# simple keras convolutional model
N, time, features = X.shape
#X = X.reshape(N, time, features, 1) # reshaping because it is expecting 4 dimensions?

filters1 = 72 #64
filters2 = 36 #32
filters3 = 18

# try out regularization next. also, save data as a csv so I don't make requests over and over
model = keras.Sequential([
    # covolve and pool
    keras.layers.Conv1D(filters1, 3, input_shape=(time, features),activation=tf.nn.relu),
    keras.layers.MaxPooling1D(pool_size=3, strides=2),
    # covolve and pool
    keras.layers.Conv1D(filters2, 3, activation=tf.nn.relu),
    keras.layers.MaxPooling1D(pool_size=3, strides=2),
    # covolve and pool
    #keras.layers.Conv1D(filters3, 3, activation=tf.nn.relu),
    #keras.layers.MaxPooling1D(pool_size=3, strides=2),
    # flatten and feed through dense layer
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(rate=.4), #.2 seems to do well, further testing needed to find optimum
    keras.layers.Dense(3, activation=tf.nn.softmax) #one for each output class
])

for layer in model.layers:
    print(layer.output_shape)

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, np.argmax(T_train, axis=1), epochs=10) # wants labels instead of indicator for some reason

test_loss, test_acc = model.evaluate(X_test, np.argmax(T_test, axis=1))

print('Test accuracy:', test_acc)
