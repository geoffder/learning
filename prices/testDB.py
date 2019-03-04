import numpy as np
import sqlite3 as sql
import pandas as pd

db = sql.connect('cryptoData.db')
cursor = db.cursor()

# a = [row for row in cursor.execute('SELECT * FROM btc_minutes')]
# print(len(a))
# print(a[0])

cursor.execute('SELECT * FROM eth_minutes GROUP BY timestamp')
a = cursor.fetchall()
df = pd.DataFrame(a)
dupes = df.duplicated([0],keep=False)
print('num dupes:', dupes.sum())

for i in range(len(df[0])-1):
    if df[0][i+1] - df[0][i] < 60:
        print('oof')
