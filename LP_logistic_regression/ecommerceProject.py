import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# note: this is how I wrote it without seeing him do it, I am leaving it as is
# since it still works. Can serve as a lesson and reminder of some functions
# and how things work, particularly pandas
def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    #print(df.head())

    # only want to work with user_actions 0 and 1

    # could drop all rows where users took other actions...
    for i in range(len(df['user_action'])):
        rows = []
        if df['user_action'][i] > 1:
            rows.append(i)
        df.drop(rows, inplace = True)
    # without reset_idex, row indices will be missing where they were removed
    df.reset_index(drop = True, inplace = True) # re-number rows to be continuous again
    #print(df.head())

    # or could change actions like begin and end purchase to 1, since the customer
    # would have to do that first to move to complete a purchase
    # df['user_action'] = np.ceil(df['user_action'] / 3.)

    # normalize non-binary data (n_products_viewed and visit_duration)
    df['n_products_viewed'] = (df['n_products_viewed'] - df['n_products_viewed'].mean())/ df['n_products_viewed'].std()
    df['visit_duration'] = (df['visit_duration'] - df['visit_duration'].mean())/ df['visit_duration'].std()

    N = len(df['user_action'])
    hot_time = np.array([[0]*df['time_of_day'][i]+[1]+[0]*(3 - df['time_of_day'][i]) for i in range(N)])
    df['12am_6am'] = hot_time[:,0]
    df['6am_12pm'] = hot_time[:,1]
    df['12pm_6pm'] = hot_time[:,2]
    df['6pm_12am'] = hot_time[:,3]
    df['ones'] = 1
    # print(df.head())

    X = df[['is_mobile', 'n_products_viewed', 'visit_duration','is_returning_visitor', '12am_6am', '6am_12pm', '12pm_6pm', '6pm_12am', 'ones']]
    labels = list(X.columns.values) # column names

    X = X.values
    T = df['user_action'].values

    return X, T, labels

X, T, labels = get_data()
N, D = X.shape # bias column already included in data processing code

# calculate Y
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# calculate derivative of obejective/cost function
def costDeriv(X, Y, T):
    return X.T @ (Y - T)

# my version of calculating cross entropy error
def crossEntropy(T, Y):
    return -(T * np.log(Y) + (1 - T)*np.log(1 - Y)).sum()
# lazy programmers way (in case mine fails, which it shouldn't)
# def crossEntropy(T, Y):
#     E = 0
#     for i in range(N):
#         if T[i] == 1:
#             E += np.log(Y[i])
#         else:
#             E += np.log(1 - Y[i])
#     return E

# find w with gradient descent
learning_rate = .001
tries = 1000

w = np.random.randn(D) / np.sqrt(D) #initialize
Y = sigmoid(X @ w)
costs = [crossEntropy(T,Y)]
for i in range(tries):
    w -= learning_rate*costDeriv(X, Y, T)
    Y = sigmoid(X @ w)
    costs.append(crossEntropy(T, Y))
    #costs.append(crossEntropy(T, np.abs(Y-.0000001)))

# how well did it do
percent = (T == np.round(Y)).sum() / N
print('----- fitting the entire set -----')
print('w:', w)
print('cross-entropy error:', costs[-1])
print('percent correct:', percent)
# plt.plot(costs)
# plt.show()

# note: LP does the test generation differently using scikit learn
# see his version for something which probably takes fewer lines
# now, do it again, but with train and test sets
train_size = int(N * .8) #80% train, 20% test
#train_idx = np.sort(np.random.randint(0, N, train_size)) # this generates duplicate numbers
train_idx = random.sample(range(N), train_size) # np.random.sample works differently, had to import random
idx = np.arange(N)
test_idx = np.delete(idx, train_idx)
print('N:', N, 'train size:', train_size, 'test size:', len(test_idx))
# use indices to define sets
Xtrain = X[train_idx,:]
Ttrain = T[train_idx]
Xtest = X[test_idx,:]
Ttest = T[test_idx]

# use gradient descent to train model
w = np.random.randn(D) / np.sqrt(D) #initialize
Ytrain = sigmoid(Xtrain @ w)
costs = [crossEntropy(Ttrain,Ytrain)]
for i in range(tries):
    w -= learning_rate*costDeriv(Xtrain, Ytrain, Ttrain)
    Ytrain = sigmoid(Xtrain @ w)
    costs.append(crossEntropy(Ttrain, Ytrain))

percent_train = (Ttrain == np.round(Ytrain)).sum() / train_size
label_weights = [labels[i] + ': ' + str(w[i]) for i in range(D)]
label_weights_block = ''
for i in range(D):
    label_weights_block += label_weights[i]
    if i < D-1:
        label_weights_block += '\n'
print('----- fitting training set -----')
print('weights:')
print(label_weights_block)
print('cross-entropy error:', costs[-1])
print('percent correct:', percent_train)
plt.plot(costs)
plt.show()

# run test data through model
Ytest = sigmoid(Xtest @ w)
percent_test = (Ttest == np.round(Ytest)).sum() / len(test_idx)
print('----- test set results-----')
print('cross-entropy error:', crossEntropy(Ttest, Ytest))
print('percent correct:', percent_test)
