import numpy as np
import sqlite3 as sql
from get_prices import pricesForDB

# connect to sql database
db = sql.connect('cryptoData.db')
cursor = db.cursor()

# load market history data
X, timestamps, tickers, metricKeys = pricesForDB()
numMetrics = len(metricKeys)

cursor.execute('SELECT MAX(timestamp) from BTC_minutes')
freshestStamp = cursor.fetchall()[0][0]

newStamps = np.argwhere(timestamps > freshestStamp).flatten()
print('number of new minutes:', newStamps.shape)
timestamps, X = timestamps[newStamps], X[newStamps, :]
for c, coin in enumerate(tickers, start=0):
    table = coin + '_minutes'
    for i in range(X.shape[0]):
        row = np.concatenate(([timestamps[i]],
                             X[i, numMetrics*c:numMetrics*c+numMetrics]))
        cursor.execute(
            'INSERT INTO ' + table + ' VALUES (?, ?, ?, ?, ?, ?, ?)', row)
db.commit()
