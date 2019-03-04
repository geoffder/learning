import numpy as np
import requests as req
import matplotlib.pyplot as plt
import time

def getMarketData():
    tickers = ['BTC', 'ETH', 'LTC']

    betTime = int(time.time() - 60*3) # unix epoch time, -three minutes

    minData = []
    metrics = {'BTC': {}, 'ETH': {}, 'LTC': {}}
    Xlist = []
    for i in range(len(tickers)):
        minData.append(req.get("https://min-api.cryptocompare.com/data/histominute?fsym="+tickers[i]+"&tsym=USD&limit=1000&toTs="+str(betTime)).json()['Data'])
        metrics[tickers[i]]['open'] = np.array([x['open'] for x in minData[i]])
        Xlist.append(metrics[tickers[i]]['open'])
        metrics[tickers[i]]['close'] = np.array([x['close'] for x in minData[i]])
        Xlist.append(metrics[tickers[i]]['close'])
        metrics[tickers[i]]['low'] = np.array([x['low'] for x in minData[i]])
        Xlist.append(metrics[tickers[i]]['low'])
        metrics[tickers[i]]['high'] = np.array([x['high'] for x in minData[i]])
        Xlist.append(metrics[tickers[i]]['high'])

    print(minData[0][0].keys())
    X = np.concatenate([Xlist], axis=1).T
    Xnorm = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)

    return X, Xnorm

if __name__ == '__main__':
    X, Xnorm = getMarketData()
    print('data shape:', X)
    #print(X)
    plt.plot(Xnorm[:,int(0*4+1)], label='BTC close norm')
    plt.plot(Xnorm[:,int(1*4+1)], label='ETH close norm')
    plt.plot(Xnorm[:,int(2*4+1)], label='LTC close norm')
    plt.legend()
    plt.show()
