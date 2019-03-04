import tensorflow as tf
from tensorflow import keras

import numpy as np
# import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from get_prices import getRacePrices

# load market history data and winner labels
# X, T = getPrices()
# alternatively load data from file
X, T = np.load('priceData.npy'), np.load('winners.npy')
# X, T = np.load('priceData_1307_1311.npy'), np.load('winners_1307_1311.npy')
# X, T = np.load('priceData_30min.npy'), np.load('winners_30min.npy')
# X, T = np.load('ETHORSE_priceData.npy'), np.load('ETHORSE_winners.npy')

print('input shapes:', X.shape, T.shape)  # T is indicator matrix

# write this class imbalance adjuster as a more generic function that I
# can plug and play more easily in other projects

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
bigN = np.max([X_btc.shape[0], X_eth.shape[0], X_ltc.shape[0]])

# roughly equalize size of each class
X_btc = np.concatenate(([X_btc]*int(np.round(bigN/X_btc.shape[0]))), axis=0)
T_btc = np.concatenate(([T_btc]*int(np.round(bigN/T_btc.shape[0]))), axis=0)
X_eth = np.concatenate(([X_eth]*int(np.round(bigN/X_eth.shape[0]))), axis=0)
T_eth = np.concatenate(([T_eth]*int(np.round(bigN/T_eth.shape[0]))), axis=0)
X_ltc = np.concatenate(([X_ltc]*int(np.round(bigN/X_ltc.shape[0]))), axis=0)
T_ltc = np.concatenate(([T_ltc]*int(np.round(bigN/T_ltc.shape[0]))), axis=0)
# print(X_btc.shape[0],X_eth.shape[0],X_ltc.shape[0])

# recombine dataset
X = np.concatenate([X_btc, X_eth, X_ltc], axis=0)
T = np.concatenate([T_btc, T_eth, T_ltc], axis=0)
print(X.shape, T.shape)

# simple keras convolutional model
N, time, num_features = X.shape

filters1 = 72  # 64
filters2 = 36  # 32
filters3 = 18
reg = .0001


def buildNetwork(time, num_features, filters1=72, filters2=36,
                 filters3=18, reg=0):
    # try out regularization next.
    model = keras.Sequential([
        # covolve and pool
        keras.layers.Conv1D(filters1, 3, input_shape=(time, num_features),
                            activation=tf.nn.relu),
        keras.layers.MaxPooling1D(pool_size=3, strides=1),
        # keras.layers.Dropout(rate=.05),
        # covolve and pool
        keras.layers.Conv1D(filters2, 3, activation=tf.nn.relu),
        keras.layers.MaxPooling1D(pool_size=3, strides=1),
        # keras.layers.Dropout(rate=.05),
        # covolve and pool
        # keras.layers.Conv1D(filters3, 3, activation=tf.nn.relu),
        # keras.layers.MaxPooling1D(pool_size=3, strides=2),

        # flatten and feed through dense layer
        keras.layers.Flatten(),
        keras.layers.Dense(36, activation=tf.nn.relu,
                           activity_regularizer=keras.regularizers.l2(reg)),
        keras.layers.Dropout(rate=.2),
        keras.layers.Dense(36, activation=tf.nn.relu,
                           activity_regularizer=keras.regularizers.l2(reg)),
        keras.layers.Dropout(rate=.2),
        # keras.layers.Dense(128, activation=tf.nn.relu,
        #                    activity_regularizer=keras.regularizers.l2(reg)),
        # keras.layers.Dropout(rate=.1),
        keras.layers.Dense(3, activation=tf.nn.softmax)  # output classes
    ])
    return model


def buildAndCompile():
    model = buildNetwork(time, num_features, filters1, filters2, filters3)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',  # targets one-hot encoded
                  metrics=['accuracy'])
    return model
# model = buildAndCompile()


# check output shapes of each layer
if 0:
    model = buildAndCompile()
    for layer in model.layers:
        print(layer.name)
        print(layer.output_shape)


def crossValidation(X, T):
    X, T = shuffle(X, T)
    sz = int(T.shape[0]/T.shape[1])
    scores = []
    for k in range(T.shape[1]):
        X_train = np.concatenate([X[:k*sz], X[k*sz+sz:]])
        T_train = np.concatenate([T[:k*sz], T[k*sz+sz:]])
        X_test = X[k*sz:k*sz+sz]
        T_test = T[k*sz:k*sz+sz]

        model = buildAndCompile()  # clear out weights
        model.fit(X_train, T_train, epochs=10)  # fit with new training set
        test_loss, test_acc = model.evaluate(X_test, T_test)  # validate
        scores.append(test_acc)

    testAcc_mean, testAcc_std = np.mean(scores), np.std(scores)
    print('mean test accuracy', testAcc_mean, 'std:', testAcc_std)
# crossValidation(X, T)


def futurePredict(X, T):
    # no shuffle, use earlier data to predict newer data
    X_train = X[:-20]
    T_train = T[:-20]
    X_test = X[-20:]
    T_test = T[-20:]
    model = buildAndCompile()
    model.fit(X_train, T_train, epochs=12)
    test_loss, test_acc = model.evaluate(X_test, T_test)

    print('test accuracy', test_acc)
# futurePredict(X, T)


'''
use date+betting_duration from API
https://bet.ethorse.com/bridge/getNonParticipatedRaces
IDEA: CMC seems to cycle updating it's endpoints in 2 and 3 minutes intervals
(when I tested BTC endpoint) what if I try training on data with variable
winner prediction lag times. e.g. winner defined by price change after
(race_length + advanceBet +/- random ~1.25mins)
'''


def predictRace(X, T):
    raceX = getRacePrices(1546469410)  # race 1310 (LTC)
    Y = []
    P = []
    for i in range(13):
        model = buildAndCompile()
        # doing this several times takes a while, but could save weights of
        # each model then load and run each of them for predictions
        print('Run:', i)
        model.fit(X[:-34], T[:-34], epochs=12)
        y = model.predict(raceX)
        Y.append(y[0, :])
        P.append(['BTC', 'ETH', 'LTC'][y[0].argmax()])

    print('predicted winner:', P)
    print('logits:', np.array(Y).round(decimals=3))
    print('BTC:', P.count('BTC'), 'ETH:', P.count('ETH'),
          'LTC:', P.count('LTC'))


predictRace(X, T)


def predictOldRaces():
    # fewer time points to match
    X, T = np.load('priceData_998.npy'), np.load('winners_998.npy')
    Xhorse = np.load('ETHORSE_priceData.npy')
    Thorse = np.load('ETHORSE_winners.npy')

    N, time, num_features = X.shape
    model = buildNetwork(time, num_features, filters1, filters2, filters3)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, T, epochs=10)
    ethorse_loss, ethorse_acc = model.evaluate(Xhorse, Thorse)
    print('ethorse test accuracy', ethorse_acc)
# predictOldRaces()
