import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

num_trials = 2000
bandit_probs = [.2, .5, .75]


class Bandit(object):
    'A slot machine!'
    def __init__(self, p):
        # probabiliity of winning
        self.p = p
        # beta parameters
        self.a = 1
        self.b = 1

    def pull(self):
        'Return bool win/loss with true probability of this bandit.'
        result = np.random.random() < self.p
        self.update(result*1)
        return result

    def sample(self):
        '''
        Return sample from current beta distribution. (Our estimate of this
        machines' win-rate.)
        '''
        return np.random.beta(self.a, self.b)

    def update(self, x):
        '''
        Update beta distributions parameters with results from pulling this
        machines' arm.
        '''
        self.a += x
        self.b += 1 - x


def plotBandits(bandits, trial):
    x = np.linspace(0, 1, 200)  # axis between 0 and 1, 200 points.
    for bandito in bandits:
        # get probability of each value of x (likelihood of each win-rate)
        y = stats.beta.pdf(x, bandito.a, bandito.b)
        plt.plot(x, y, label='real p: %.4f' % bandito.p)
    plt.xlabel('Win Rate')
    plt.title('Bandit Distributions after %s trials' % trial)
    plt.legend()
    plt.show()


def non_adaptive():
    '''
    Test all bandits equally, to get estimates of their win-rates.
    Not following any adaptive strategy.
    '''
    bandits = [Bandit(p) for p in bandit_probs]

    # trial numbers at which we'll take a look at the win-rate estimates
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1500, 1999]

    for i in range(num_trials):
        for bandito in bandits:
            bandito.pull()

        if i in sample_points:
            plotBandits(bandits, i)


def play_to_win():
    '''
    Perform Bayesian A/B testing, with strategy to maximize win-rate, pulling
    the bandit with the highest estimated win-rate on each trial.
    '''
    bandits = [Bandit(p) for p in bandit_probs]

    # trial numbers at which we'll take a look at the win-rate estimates
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1500, 1999]

    for i in range(num_trials):
        best_bandit = np.argmax([bandito.sample() for bandito in bandits])
        bandits[best_bandit].pull()

        if i in sample_points:
            plotBandits(bandits, i)


if __name__ == '__main__':
    play_to_win()
    # non_adaptive()
