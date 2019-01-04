import numpy as np
import sqlite3 as sql
from get_prices import pricesForDB

# connect to sql database
db = sql.connect('cryptoData.db')
cursor = db.cursor()

# load market history data
X, timestamps, tickers, metricKeys = pricesForDB()
numMetrics = len(metricKeys)

for c, coin in enumerate(tickers, start=0):
    table = coin + '_minutes'
    for i in range(X.shape[0]):
        row = np.concatenate(([timestamps[i]],X[i, numMetrics*c:numMetrics*c+numMetrics]))
        cursor.execute(
            'INSERT INTO ' + table + ' VALUES (?, ?, ?, ?, ?, ?, ?)', row)
db.commit()

# this is a start, but now I have to write it so time stamps that already
# exist are excluded. SELECT timestamp column and use that to trim input data
