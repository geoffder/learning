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
print('input shapes:', X.shape, T.shape) # T is indicator matrix

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

# simple keras convolutional model
N, time, features = X.shape

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
    keras.layers.Dropout(rate=.2), #.2 seems to do well, further testing needed to find optimum
    keras.layers.Dense(3, activation=tf.nn.softmax) #one for each output class
])

for layer in model.layers:
    print(layer.output_shape)

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def crossValidation(model, X, T):
    X, T = shuffle(X, T)
    sz = int(T.shape[0]/T.shape[1])
    scores = []
    for k in range(T.shape[1]):
        X_train = np.concatenate([X[:k*sz], X[k*sz+sz:]])
        T_train = np.concatenate([T[:k*sz], T[k*sz+sz:]])
        X_test = X[k*sz:k*sz+sz]
        T_test = T[k*sz:k*sz+sz]

        model.fit(X_train, np.argmax(T_train, axis=1), epochs=10)
        test_loss, test_acc = model.evaluate(X_test, np.argmax(T_test, axis=1))
        scores.append(test_acc)

    return np.mean(scores), np.std(scores)

testAcc_mean, testAcc_std = crossValidation(model, X, T)
print('mean test accuracy', testAcc_mean, 'std:', testAcc_std)
