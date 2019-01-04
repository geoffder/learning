import numpy as np
import sqlite3 as sql

db = sql.connect('cryptoData.db')
cursor = db.cursor()

a = [row for row in cursor.execute('SELECT * FROM btc_minutes')]
print(len(a))
print(a[0])
