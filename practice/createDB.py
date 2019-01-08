import numpy as np
import sqlite3 as sql
from get_prices import getPrices

# connect to sql database
db = sql.connect('cryptoData2.db')
cursor = db.cursor()

# create tables for market data
cursor.execute(
    '''
    CREATE TABLE BTC_minutes
    (timestamp real, open real, close real, low real,
    high real, volumefrom real, volumeto real)
    '''
)
cursor.execute(
    '''
    CREATE TABLE ETH_minutes
    (timestamp real, open real, close real, low real,
    high real, volumefrom real, volumeto real)
    '''
)
cursor.execute(
    '''
    CREATE TABLE LTC_minutes
    (timestamp real, open real, close real, low real,
    high real, volumefrom real, volumeto real)
    '''
)
db.commit()
db.close()
