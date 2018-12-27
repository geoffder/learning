import numpy as np
import matplotlib.pyplot as plt

import requests as req
#from requests_futures.sessions import FuturesSession
#from concurrent.futures import wait

import time


metricKeys = ['open', 'close', 'low', 'high', 'volumefrom', 'volumeto']

def getMarketData():
    # build endpoints
    tickers = ['BTC', 'ETH', 'LTC']
    betTime = int(time.time() - 60*3) # unix epoch time, -three minutes
    # 2000 is max number of minutes at a time, get more by repeating with different &toTs=
    endpoints = ["https://min-api.cryptocompare.com/data/histominute?fsym=%s&tsym=USD&limit=2000&toTs=%d" % (s, betTime) for s in tickers]
    headers = {'authorization': 'e35795772b742f4c32495080678cdab098817bcab591b2f44d0aae61c2d32ef3'}
    features = []
    for i in range(len(tickers)):
        minData = req.get(endpoints[i], headers=headers).json()['Data']
        for key in metricKeys:
            features.append(np.array([x[key] for x in minData]))

    X = np.concatenate([features], axis=1).T
    Xnorm = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)

    return X, Xnorm

def createSamples(X):
    length = 1000 # minutes of data per sample (leading up to prediction)

    Xlist = []
    Ylist = []
    encode = [[1,0,0], [0,1,0], [0,0,1]]
    for t in range(1000, X.shape[0]-70, 5):
        Xlist.append(X[t-1000:t,:])
        change = []
        for i in range(3):
            priceStart = Xnorm[t+4,int(i*len(metricKeys)+1)]
            priceEnd = Xnorm[t+64,int(i*len(metricKeys)+1)]
            change.append((priceEnd - priceStart)/priceStart)
        Ylist.append(encode[np.argmax(change)])

    Y =  np.concatenate([Ylist], axis=0)
    return Xlist, Y

if __name__ == '__main__':
    X, Xnorm = getMarketData()
    print('X dataset shape:', X.shape)
    #print(X)
    if(0):
        plt.plot(Xnorm[:,int(0*len(metricKeys)+1)], label='BTC close norm')
        plt.plot(Xnorm[:,int(1*len(metricKeys)+1)], label='ETH close norm')
        plt.plot(Xnorm[:,int(2*len(metricKeys)+1)], label='LTC close norm')
        plt.legend()
        plt.show()
    Xlist, Y = createSamples(Xnorm)
    print('Xlist length:', len(Xlist), '; X sample shape:', Xlist[0].shape, '; Y shape:', Y.shape)
