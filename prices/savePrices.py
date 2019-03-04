import numpy as np
from get_prices import getPrices

# load market history data and winner labels
X, T = getPrices()
np.save("priceData.npy", X)
np.save("winners.npy", T)
