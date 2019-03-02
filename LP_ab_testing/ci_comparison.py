import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

'''
Demonstration of central limit theorem (that a variable calculated from
random variables is a random variable). We see that that the Gaussian
approximation of the CTR (with parameters mean and std) closely mirrors that of
the Bayesian posterior (itself, a probability distributio of possible CTRs).
'''

N = 501  # number of iterations
true_ctr = .5  # actual click-through rate
a, b = 1, 1  # beta priors. (1, 1) is a uniform distributions
plot_indices = [10, 20, 30, 50, 100, 200, 500]
data = np.empty(N)

for i in range(N):
    data[i] = 1 if np.random.random() > true_ctr else 0

    a += data[i]
    b += 1 - data[i]

    if i in plot_indices:
        p = data[:i].mean()  # mean of all data so far
        n = i + 1  # zero indexing
        std = np.sqrt(p*(1-p)/n)
        x = np.linspace(0, 1, 200)  # check probabilities of these CTRs
        gaussian = norm.pdf(x, loc=p, scale=std)
        posterior = beta.pdf(x, a, b)
        plt.plot(x, gaussian, label='Gaussian Approximation')
        plt.plot(x, posterior, label='Beta Posterior')
        plt.title('Trial: %d' % i)
        plt.xlabel('Click Through Rate')
        plt.legend()
        plt.show()
