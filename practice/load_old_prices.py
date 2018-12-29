import numpy as np
import json

raceData = json.loads(open('raceData.json').read())
minuteData = json.loads(open('minuteData.json').read())

tickers = ['BTC', 'ETH', 'LTC']
metricKeys = ['open', 'close', 'low', 'high', 'volumefrom', 'volumeto']

lastRace=1
raceNumbers = [int(x['race_number']) for x in raceData]
for i in range(len(raceData)):
    if raceNumbers[i] > lastRace:
        lastRace = raceNumbers[i]
startRace = lastRace-380;

raceIdxs = []
for r in range(startRace, lastRace):
    try:
        idx = raceNumbers.index(r) # index passes an error if the given index does not appear
        if raceData[idx]['race_duration'] == '3600' and len(minuteData['BTC']['min'][idx]) != 0:
            raceIdxs.append(idx)
    except:
        #print('missing race')
        pass

Xlist = []
#for i in range(len(minuteData['BTC']['min'])):
for idx in raceIdxs:
    features = []
    for ticker in tickers:
        for key in metricKeys:
            features.append(np.array([x[key] for x in minuteData[ticker]['min'][idx]]))

    sample = np.concatenate([features], axis=1).T
    sampleNorm = (sample - sample.mean(axis=0, keepdims=True)) / sample.std(axis=0, keepdims=True)
    Xlist.append(np.array(sampleNorm[:-3])) # 3 minutes before race

print(len(Xlist),Xlist[0].shape)

# turn race winners in to indicator target matrix
Tlist = []
encode = {'BTC': [1,0,0], 'ETH': [0,1,0], 'LTC': [0,0,1]}
Tlist = [encode[raceData[idx]['winner']] for idx in raceIdxs]

T =  np.concatenate([Tlist], axis=0)
X3d = np.concatenate([Xlist], axis=2)

print('X shape:', X3d.shape, 'T shape:', T.shape)

np.save("ETHORSE_priceData.npy", X3d)
np.save("ETHORSE_winners.npy", T)
