import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # don't need but maybe use to get familiar with API

from sklearn.utils import shuffle

df = pd.read_csv('fer2013.csv')
#print(df.columns.values) # ['emotion' 'pixels' 'Usage']
# emotions = []
# for ele in df['emotion'].values:
#     if not ele in emotions:
#         emotions.append(ele)
# print(np.sort(emotions)) # [0 to 6]
# usages = []
# for ele in df['Usage'].values:
#     if not ele in usages:
#         usages.append(ele)
# print(usages)

# emotion: ints 0 through 6
# pixels: string object of ints seperated by spaces
# Usage: ['Training, 'PublicTest', 'PrivateTest']

#steps
# emotions are the classes, 7 of them. Only work on 0 and 1, so it is binary
# there are way more 0s than 1s though, need to build the training and test set so that n for each is equal
# break the pixels string up, and make each pixel a feature. normalize the values
# don't really care about the Usage I don't think. I'll just shuffle to make my train and test sets
