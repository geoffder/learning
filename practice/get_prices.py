import numpy as np
import matplotlib.pyplot as plt

import requests as req
import asyncio

import time

# global coin labels
tickers = ['BTC', 'ETH', 'LTC']
metricKeys = ['open', 'close', 'low', 'high', 'volumefrom', 'volumeto']

async def makeRequests(endpoints, headers):
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(
            None,
            req.get,
            endpoints[i]
        )
        for i in range(len(endpoints))
    ]
    minData = []
    for response in await asyncio.gather(*futures): # in order of requests?
        minData.append(response.json()['Data'])

    return minData

def buildRequests():
    # build endpoints
    # get it on the minute to facilitate database storage
    lastTime = int(time.time() - 60*4 - time.time() % 60) # unix epoch time
    toTime = [int(lastTime - i*(60*2000)) for i in range(5)]
    # 2000 is max number of minutes at a time, get more by repeating with different #&toTs=
    endpoints = ["https://min-api.cryptocompare.com/data/histominute?fsym=%s&tsym=USD&limit=2000&toTs=%d" % (
        s, t) for s in tickers for t in toTime]
    headers = {'authorization': 'e35795772b742f4c32495080678cdab098817bcab591b2f44d0aae61c2d32ef3'}
    return endpoints, headers, lastTime

def sortData(minData, batches):
    # first bring all of the time blocks for each coin together
    coinData = [[],[],[]]
    for i in range(len(coinData)):
        for j in range(batches):
            coinData[i] += minData[int(i*j + j)]

    features = []
    for coin in coinData:
        for key in metricKeys:
            features.append(np.array([x[key] for x in coin]))

    X = np.concatenate([features], axis=1).T
    Xnorm = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)

    return X, Xnorm

def createSamples(X):
    # sampling variables
    hist = 1000 # minutes of data per sample (leading up to prediction)
    raceLen = 60
    advanceBet = 14 # number of minutes in advance (?5 minute offset from new race prediction?)
    jitterRange = 6 #3, 10 have also had some success


    Xlist = []
    Ylist = []
    encode = [[1,0,0], [0,1,0], [0,0,1]]
    # start at minute 1000 (each race has 1000 minutes history data)
    # step is number of minutes between sanples
    for t in range(hist, X.shape[0]-raceLen-advanceBet-10, 30): # is there a benefit of more artificial race samples?
        Xlist.append(X[t-hist:t,:])
        change = []
        # jitter = ((np.random.random(2)-.5) * jitterRange).astype(np.int32)
        for i in range(3):
            jitter = ((np.random.random(2)-.5) * jitterRange).astype(np.int32)
            priceStart = X[t+advanceBet+jitter[0], int(i*len(metricKeys)+1)]
            priceEnd = X[t+raceLen+advanceBet+jitter[1], int(i*len(metricKeys)+1)]
            change.append((priceEnd - priceStart)/priceStart)
        Ylist.append(encode[np.argmax(change)])

    Y =  np.concatenate([Ylist], axis=0)
    X3d = np.concatenate([Xlist], axis=2)
    return X3d, Y

# do everything and return X and target labels
def getPrices():
    endpoints, headers, _ = buildRequests()
    loop = asyncio.get_event_loop()
    minData = loop.run_until_complete(makeRequests(endpoints, headers))

    X, Xnorm = sortData(minData, 5)
    X3d, Y = createSamples(Xnorm)

    return X3d, Y

# get price data leading up to a particular race, takes unix epoch start time
def getRacePrices(raceStart):
    # build endpoints
    lastTime = int(raceStart - 60*14) # unix epoch time,
    endpoints = [
        "https://min-api.cryptocompare.com/data/histominute?fsym=%s&tsym=USD&limit=999&toTs=%d" % (
        s, lastTime) for s in tickers]
    headers = {'authorization': 'e35795772b742f4c32495080678cdab098817bcab591b2f44d0aae61c2d32ef3'}

    loop = asyncio.get_event_loop()
    minData = loop.run_until_complete(makeRequests(endpoints, headers))

    _, Xnorm = sortData(minData, 1)
    X3d = np.concatenate([[Xnorm, Xnorm]], axis=2)
    return X3d

# do everything and return X and target labels
def pricesForDB():
    # get minute data from crypto-compare
    endpoints, headers, lastTime = buildRequests()
    loop = asyncio.get_event_loop()
    minData = loop.run_until_complete(makeRequests(endpoints, headers))

    X, _ = sortData(minData, 5)
    firstTime = int(lastTime - 60*(X.shape[0]-1))
    timestamps = np.arange(firstTime, lastTime+60, 60)
    return X, timestamps, tickers, metricKeys

if __name__ == '__main__':
    endpoints, headers = buildRequests()
    loop = asyncio.get_event_loop()
    minData = loop.run_until_complete(makeRequests(endpoints, headers))
    # check to make sure that the responses are in order
    print('first prices:', minData[0][0]['close'],minData[1][0]['close'],minData[2][0]['close'])
    X, Xnorm = sortData(minData)
    print('X dataset shape:', X.shape)
    #print(X)
    if(0):
        plt.plot(Xnorm[:,int(0*len(metricKeys)+1)], label='BTC close norm')
        plt.plot(Xnorm[:,int(1*len(metricKeys)+1)], label='ETH close norm')
        plt.plot(Xnorm[:,int(2*len(metricKeys)+1)], label='LTC close norm')
        plt.legend()
        plt.show()
    X3d, Y = createSamples(Xnorm)
    print('X3d (sample, minutes, features)', X3d.shape, '; Y shape:', Y.shape)
