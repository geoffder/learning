import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from get_prices import getPrices, getRacePrices

# load market history data and winner labels
#X, T = getPrices()
# alternatively load data from file
X, T = np.load('priceData.npy'), np.load('winners.npy')
#X, T = np.load('priceData_1307_1311.npy'), np.load('winners_1307_1311.npy')
#X, T = np.load('priceData_30min.npy'), np.load('winners_30min.npy')
#X, T = np.load('ETHORSE_priceData.npy'), np.load('ETHORSE_winners.npy')

print('input shapes:', X.shape, T.shape) # T is indicator matrix

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
N, time, num_features = X.shape

filters1 = 72 #64
filters2 = 36 #32
filters3 = 18
reg = .0001

def buildNetwork(time, num_features, filters1=72, filters2=36, filters3=18, reg=0):
    # try out regularization next.
    model = keras.Sequential([
        # covolve and pool
        keras.layers.Conv1D(filters1, 3, input_shape=(time, num_features),activation=tf.nn.relu),
        keras.layers.MaxPooling1D(pool_size=3, strides=1), # strides down from 2 to 1
        # keras.layers.Dropout(rate=.05),
        # covolve and pool
        keras.layers.Conv1D(filters2, 3, activation=tf.nn.relu),
        keras.layers.MaxPooling1D(pool_size=3, strides=1),
        # keras.layers.Dropout(rate=.05),
        # covolve and pool
        #keras.layers.Conv1D(filters3, 3, activation=tf.nn.relu),
        #keras.layers.MaxPooling1D(pool_size=3, strides=2),

        # flatten and feed through dense layer
        keras.layers.Flatten(),
        keras.layers.Dense(36, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(reg)), #128
        keras.layers.Dropout(rate=.2),
        keras.layers.Dense(36, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(reg)),
        keras.layers.Dropout(rate=.2),
        # keras.layers.Dense(128, activation=tf.nn.relu, activity_regularizer=keras.regularizers.l2(reg)),
        # keras.layers.Dropout(rate=.1),
        keras.layers.Dense(3, activation=tf.nn.softmax) #one for each output class
    ])
    return model

# check output shapes of each layer
if 0:
    for layer in model.layers:
        print(layer.name)
        print(layer.output_shape)

def buildAndCompile():
    model = buildNetwork(time, num_features, filters1, filters2, filters3)
    model.compile(optimizer=tf.train.AdamOptimizer(),
              #loss='sparse_categorical_crossentropy', # targets as integers
              loss='categorical_crossentropy', # targets one-hot encoded
              metrics=['accuracy'])
    return model
# model = buildAndCompile()

def crossValidation(X, T):
    X, T = shuffle(X, T)
    sz = int(T.shape[0]/T.shape[1])
    scores = []
    for k in range(T.shape[1]):
        X_train = np.concatenate([X[:k*sz], X[k*sz+sz:]])
        T_train = np.concatenate([T[:k*sz], T[k*sz+sz:]])
        X_test = X[k*sz:k*sz+sz]
        T_test = T[k*sz:k*sz+sz]

        model = buildAndCompile() # clear out weights
        model.fit(X_train, T_train, epochs=10) # fit with new training set
        test_loss, test_acc = model.evaluate(X_test, T_test) # fit against test set k
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

def predictRace(X, T):
    # use date+betting_duration from API https://bet.ethorse.com/bridge/getNonParticipatedRaces
    # note that the scheduled race start time is around a minute late vs earliest oracle time stamp
    # raceX = getRacePrices(1546448391) # race 1307 (BTC) # (oracle start: 1546448343) solid
    # raceX = getRacePrices(1546455607) #race 1308 (ETH) # (oracle start: 1546455543) correct but variable
    raceX = getRacePrices(1546469410) #race 1310 (LTC)  # (oracle start: 1546469343) very wrong (hard BTC)
    # raceX = getRacePrices(1546501807) #race 1311 (BTC)
    # NOTE: the result can change from run to run, I think dropout might have caused this
    #       without it, sometimes the model seems to get stuck in a rut and not reach 100%
    #       accuracy on the train set, but seems to be more reliable as long as ~1.0 was achieved in training
    #       Options: try out lower dropout, or simply refresh model and re-fit if training was not a success.
    #       (was running .2 dropout when I realized it might be the problem)
    #       Also trying additional training epochs. Might have improved hit rate.

    # MORE NOTES: predicting races with minutes older than training may be required (CMC sketchiness)
    #       appears that CMC can be +/- 5 minutes off. #1308 does well with 5 mins older data than training
    #       #1310 does well with the same or 5 minutes earlier than training
    #       e.g. Train model using 69 minutes from data to prediction (9 mins before 60m race)
    #                   #1308: make prediction 14 minutes before race (+5 prediction lag)
    #                   #1310: make prediction 9 (or 4 = better) minutes before race (-5 prediction lag)
    #       Challenge: can I predict which one it will be by comparing CMC to CryptoCompare data?
    # IDEA: CMC seems to cycle updating it's endpoints in 2 and 3 minutes intervals (when I tested BTC endpoint)
    #       what if I try training on data with variable winner prediction lag times
    #       e.g. winner defined by price change after (race_length + advanceBet +/- random ~1.25mins)
    Y = []
    P = []
    for i in range(13):
        model = buildAndCompile()
        # doing this several times takes a while, but could save weights of
        # each model then load and run each of them for predictions only at betting time
        print('Run:', i)
        model.fit(X[:-34], T[:-34], epochs=12)
        y = model.predict(raceX)
        Y.append(y[0,:])
        P.append(['BTC', 'ETH', 'LTC'][y[0].argmax()])

    print('predicted winner:', P)
    print('logits:', np.array(Y).round(decimals=3))
    print('BTC:', P.count('BTC'), 'ETH:', P.count('ETH'), 'LTC:', P.count('LTC'))
predictRace(X, T)




def predictOldRaces():
    X, T = np.load('priceData_998.npy'), np.load('winners_998.npy') # fewer time points to match
    Xhorse, Thorse = np.load('ETHORSE_priceData.npy'), np.load('ETHORSE_winners.npy')

    # cant change attribute after creation in this way, need to write this code differently
    # need to use these data sets from the start if I want to do this
    #model.get_layer('conv1d').input_shape = (None, 998, 18)

    # just rebuild a new network with the correct input shape
    N, time, num_features = X.shape
    model = buildNetwork(time, num_features, filters1, filters2, filters3)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, T, epochs=10)
    ethorse_loss, ethorse_acc = model.evaluate(Xhorse, Thorse)
    print('ethorse test accuracy', ethorse_acc)

#predictOldRaces()
