import numpy as np
import json

raceData = json.loads(open('raceData.json').read())
minuteData = json.loads(open('minuteData.json').read())

# relevant block from minuteTester.js
# metrics[coinTickers[j]].min.high = priceData[coinTickers[j]].min[i].map(x => x.high);
# metrics[coinTickers[j]].min.low = priceData[coinTickers[j]].min[i].map(x => x.low);
# metrics[coinTickers[j]].min.close = priceData[coinTickers[j]].min[i].map(x => x.close);
# metrics[coinTickers[j]].min.open = priceData[coinTickers[j]].min[i].map(x => x.open);
# metrics[coinTickers[j]].min.vol = priceData[coinTickers[j]].min[i].map(x => x.volumeto);

# race winner for each race is stored like so
#raceData[i]['winner'] = 'BTC' / 'ETH' / 'LTC'
