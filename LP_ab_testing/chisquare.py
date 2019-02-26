import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class DataGenerator(object):
    'Takes probability of two groups. Use next() to generate CTR like data.'
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def next(self):
        click1 = 1 if np.random.random() < self.p1 else 0
        click2 = 1 if np.random.random() < self.p2 else 0
        return [click1, click2]


def buildTable(A, B):
    table = np.zeros((2, 2))
    table[0, :] = [A.sum(), A.shape - A.sum()]
    table[1, :] = [B.sum(), B.shape - B.sum()]
    return table


def get_p_value(T):
    det = T[0, 0]*T[1, 1] - T[0, 1]*T[1, 0]
    c2 = det/T[0].sum() * det/T[1].sum() * T.sum()/T[:, 0].sum()/T[:, 1].sum()
    p = 1 - stats.chi2.cdf(x=c2, df=1)
    return p


def run_batch(p1, p2, N):
    'Generate fresh data (N samples) in one go and analyze.'
    data = DataGenerator(p1, p2)
    A, B = np.array([data.next() for _ in range(N)]).T
    table = buildTable(A, B)
    p_value = get_p_value(table)

    return A.mean(), B.mean(), p_value


def run_experiment(p1, p2, N):
    '''
    Iteratively add samples and test as N grows. Less noisy than run_batch as
    N increases, since it is the same data at each step with one added sample,
    rather than an entirely new dataset. Can see why run_batch is so noisy,
    each run of this function follows a wildly different trajectory, sometimes
    never achieving significant p-values.
    '''
    data = DataGenerator(p1, p2)
    table = np.zeros((2, 2)).astype(np.float32)
    p_values = np.empty(N)

    for i in range(N):
        c1, c2 = data.next()
        table[0, c1] += 1  # add to either yes or no cell
        table[1, c2] += 1
        if i < 10:
            p_values[i] = None  # too low N may cause div by zero
        else:
            p_values[i] = get_p_value(table)

    plt.plot(p_values)
    plt.plot(np.ones(N)*.05)  # alpha line
    plt.show()


def main():
    minN = 20
    maxN = 10000  # highest N experiment to run
    p1, p2 = .1, .11  # true probabilities of each group

    meanDeltas = []  # diffs between calculated rates and input probabilities
    p_values = []
    for N in range(minN, maxN+1, 10):
        A_mean, B_mean, p_value = run_batch(p1, p2, N)
        meanDeltas.append([np.abs(A_mean - p1), np.abs(B_mean - p2)])
        p_values.append(p_value)

    meanDeltas = np.array(meanDeltas)
    print(meanDeltas.shape)
    x_axis = np.arange(minN, maxN+1, 10)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(x_axis, meanDeltas[:, 0])
    axes[0].plot(x_axis, meanDeltas[:, 1])
    axes[0].set_ylabel('rates vs probabilities (absolute diff)')
    axes[0].set_xlabel('N')
    axes[1].plot(x_axis, p_values)
    axes[1].set_ylabel('chi-square p-value')
    axes[1].set_xlabel('N')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    run_experiment(.1, .11, 20000)
