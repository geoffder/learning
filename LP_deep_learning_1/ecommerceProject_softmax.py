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

    X = df[['is_mobile', 'n_products_viewed', 'visit_duration','is_returning_visitor', '12am_6am', '6am_12pm', '12pm_6pm', '6pm_12am']]
    labels = list(X.columns.values) # column names

    X = X.values
    T = df['user_action'].values
    Thot = np.zeros((T.shape[0], int(T.max()+1)))
    Thot[np.arange(T.shape[0]), T] = 1

    return X, T, Thot, labels

X, T, Thot, labels = get_data()
N, D = X.shape
K = Thot.shape[1]

# calculate Y
def forward(X, w, b):
    return softmax(X @ w + b)

# calculate derivative of obejective/cost function
def costDeriv(X, Y, T):
    return X.T @ (Y - T)

# my version of calculating binary cross entropy error
def crossEntropy(T, Y): # binary
    return -(T * np.log(Y) + (1 - T)*np.log(1 - Y)).sum()

def cost(T, Y): # multiclass
    return (T * np.log(Y)).sum()

def softmax(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)

# find w with gradient descent
learning_rate = .0001
tries = 3000

w = np.random.randn(D,K)
b = np.random.randn(K)

costs = []
for i in range(tries):
    Y = forward(X, w, b)
    w -= learning_rate * costDeriv(X, Y, Thot)
    b -= learning_rate * (Y - Thot).sum(axis=0)
    costs.append(cost(Thot, Y))

# how well did it do
percent = (T == np.argmax(Y, axis=1)).sum() / N
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
Ttrain_hot = Thot[train_idx,:]
Xtest = X[test_idx,:]
Ttest = T[test_idx]
Ttest_hot = Thot[test_idx,:]
# use gradient descent to train model
w = np.random.randn(D,K)
b = np.zeros(K) #initialize

costs = []
for i in range(tries):
    Ytrain = forward(Xtrain, w, b)
    w -= learning_rate*costDeriv(Xtrain, Ytrain, Ttrain_hot)
    b -= learning_rate * (Ytrain - Ttrain_hot).sum(axis=0)
    costs.append(crossEntropy(Ttrain_hot, Ytrain))

percent_train = (Ttrain == np.argmax(Ytrain, axis=1)).sum() / train_size
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
Ytest = forward(Xtest, w, b)
percent_test = (Ttest == np.argmax(Ytest, axis=1)).sum() / len(test_idx)
print('----- test set results-----')
print('cross-entropy error:', crossEntropy(Ttest_hot, Ytest))
print('percent correct:', percent_test)
