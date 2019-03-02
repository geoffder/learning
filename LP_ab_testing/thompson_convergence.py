import numpy as np
import matplotlib.pyplot as plt
from bayesian_bandit import Bandit

'''
Demonstration that the actual win-date (data) converges on the highest win-rate
option available (best possible) result automatically and quickly. No test
required to understand that a convergence has been reached.
'''


def run_experiment(p1, p2, p3, N):
    'args: probabilities for 3 bandits and number of experiments.'
    bandits = [Bandit(p1), Bandit(p2), Bandit(p3)]

    data = np.empty(N)

    for i in range(N):
        best_bandito = np.argmax([bandito.sample() for bandito in bandits])
        result = bandits[best_bandito].pull()
        data[i] = result

    cumulative_avg_ctr = np.cumsum(data) / (np.arange(N) + 1)
    plt.plot(cumulative_avg_ctr)
    plt.plot(np.ones(N)*p1, '--')
    plt.plot(np.ones(N)*p2, '-.')
    plt.plot(np.ones(N)*p3, ':')
    plt.ylim((0, .6))
    plt.xscale('log')
    plt.xlabel('Number of Trials')
    plt.show()


if __name__ == '__main__':
    run_experiment(.2, .25, .3, 100000)
