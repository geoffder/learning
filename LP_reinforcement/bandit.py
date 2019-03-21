import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


class Bandit(object):
    "A slot machine!"
    def __init__(self, p, initial_mu=0):
        # probabiliity of winning
        self.p = p
        # sample mean parameters
        self.mu = initial_mu  # mean
        self.N = 0  # samples collected
        # beta parameters (Bayesian)
        self.a = 1
        self.b = 1

    def pull(self):
        "Return bool win/loss with true probability of this bandit."
        result = np.random.random() < self.p
        self.betaUpdate(result*1)
        self.updateMean(result*1)
        return result

    def updateMean(self, x):
        "Update sample mean."
        self.N += 1
        self.mu = (1 - 1/self.N)*self.mu + x/self.N

    def betaSample(self):
        """
        Return sample from current beta distribution. (Our estimate of this
        machines' win-rate.)
        """
        return np.random.beta(self.a, self.b)

    def betaUpdate(self, x):
        """
        Update beta distributions parameters with results from pulling this
        machines' arm.
        """
        self.a += x
        self.b += 1 - x


def plotBetas(bandits, trial):
    x = np.linspace(0, 1, 200)  # axis between 0 and 1, 200 points.
    for bandito in bandits:
        # get probability of each value of x (likelihood of each win-rate)
        y = beta.pdf(x, bandito.a, bandito.b)
        plt.plot(x, y, label='real p: %.4f' % bandito.p)
    plt.xlabel('Win Rate')
    plt.title('Bandit Distributions after %s trials' % trial)
    plt.legend()
    plt.show()


def play_to_win(n_trials=2000, bandit_probs=[.2, .5, .75], strategy='Bayesian',
                epsilon=.05, optim_mu=1, show_fig=False):
    """
    Perform Bayesian A/B testing, with strategy to maximize win-rate, pulling
    the bandit with the highest estimated win-rate on each trial.
    """
    print("Playing using %s strategy..." % strategy)
    print("True Bandit probabilities:", bandit_probs)
    if strategy == 'Optimistic':
        bandits = [Bandit(p, initial_mu=optim_mu) for p in bandit_probs]
    else:
        bandits = [Bandit(p) for p in bandit_probs]

    # trial numbers at which we'll take a look at the win-rate estimates
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1500, 1999]

    reward = 0
    for i in range(n_trials):
        if strategy == 'Greedy':
            # epsilon gives chance to randomly choose bandit, otherwise go
            # with the current estimated best option (highest sample mean)
            if np.random.random() > epsilon:
                best_bandit = np.argmax([bandito.mu for bandito in bandits])
            else:
                best_bandit = np.random.choice(np.arange(len(bandits)))
        elif strategy == 'Optimistic':
            # all greed, all the time
            best_bandit = np.argmax([bandito.mu for bandito in bandits])
        elif strategy == 'UCB1':
            if not i:  # first trial, no data yet
                best_bandit = np.random.choice(np.arange(len(bandits)))
            else:
                # epsilon is dynamic adjustment of sample mean.
                # epsilon = sqrt(ln(total_N)/option_N)
                best_bandit = np.argmax(
                    [b.mu + np.sqrt(2*np.log(i)/(b.N+.00001))
                     for b in bandits])
        elif strategy == 'Bayesian':
            # draw from beta distributions to choose bandit to pull
            best_bandit = np.argmax([b.betaSample() for b in bandits])

        reward += bandits[best_bandit].pull()

        if i in sample_points:
            if strategy == 'Bayesian' and show_fig:
                plotBetas(bandits, i)
            print("Bandit means and Ns at trial %d:" % i)
            print([(bandito.mu, bandito.N) for bandito in bandits])
            print('Total reward so far (trial %d): %d' % (i, reward))
        elif i > sample_points[-1] and (i+1) % 1000 == 0:
            print('Total reward so far (trial %d): %d' % (i, reward))


if __name__ == '__main__':
    play_to_win(n_trials=10000, bandit_probs=[.2, .5, .75],
                strategy='UCB1', epsilon=.05, show_fig=True)
