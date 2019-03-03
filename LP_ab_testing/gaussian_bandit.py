import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class Bandit(object):
    'A Gaussian machine! Continuous reward values, rather than bool win/loss.'
    def __init__(self, params):
        # actual probabilistic reward parameters (Gaussian Likelihood)
        self.mu = params[0]
        self.tau = params[1]
        # Gaussian mu parameters (prior is N(0, 1))
        self.m = 0
        self.lambda_ = 1
        self.sum = 0  # running sum of samples
        self.N = 0  # count of samples

    def pull(self):
        'Return reward value using with true parameters of this bandit.'
        result = np.random.randn()/np.sqrt(self.tau) + self.mu
        self.update(result)
        return result

    def sample(self):
        '''
        Return sample from current estimated Gaussian distribution.
        (Our estimate of this machines' reward value.)
        '''
        return np.random.randn()/np.sqrt(self.lambda_) + self.m

    def update(self, x):
        '''
        Update Gaussian parameters with result (continuous variable) from
        pulling this machines' arm.
        '''
        self.N += 1  # increment sample size
        self.sum += x  # running sum of all samples
        # update posterior lambda (precision).
        self.lambda_ += self.tau  # assuming variance fixed, not using a prior.
        # update posterior m (mean).
        # self.m = (self.m0*self.lambda0 + self.tau*self.sum)/self.lamb
        # since prior m0 is zero, the term is always zero, so...
        self.m = (self.tau*self.sum)/self.lambda_


def plotBandits(bandits, trial):
    x = np.linspace(0, 1, 200)  # axis between 0 and 1, 200 points.
    for bandito in bandits:
        # get probability of each value of x (likelihood of each win-rate)
        y = norm.pdf(x, loc=bandito.m, scale=np.sqrt(1/bandito.lambda_))
        plt.plot(x, y, label='real mu: %.4f; real tau: %.4f'
                 % (bandito.mu, bandito.tau))
    plt.xlabel('Estimated Reward Value')
    plt.title('Bandit Distributions after %s trials' % trial)
    plt.legend()
    plt.show()


def play_to_win():
    '''
    Perform Bayesian A/B testing, with strategy to maximize win-rate, pulling
    the bandit with the highest estimated win-rate on each trial.
    '''
    bandits = [Bandit(p) for p in bandit_params]

    # trial numbers at which we'll take a look at the win-rate estimates
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1500, 1999]

    for i in range(num_trials):
        best_bandit = np.argmax([bandito.sample() for bandito in bandits])
        bandits[best_bandit].pull()

        if i in sample_points:
            print('b1 N=%d times; b2 N=%d times; b3 N=%d'
                  % (bandits[0].N, bandits[1].N, bandits[2].N))
            plotBandits(bandits, i)


if __name__ == '__main__':
    num_trials = 2000
    bandit_params = [(.2, 1), (.5, 1), (.8, 1)]
    play_to_win()
