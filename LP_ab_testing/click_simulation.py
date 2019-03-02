import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt

'''
Simulation of a website serving up advertisements using an adaptive strategy.
Basically the same as the combo of server.py and client.py, but much faster,
with plots to boot.
'''


class Bandit:
    def __init__(self, name):
        self.name = name
        # initialize beta parameters
        self.a = 1
        self.b = 1

    def sample(self):
        return np.random.beta(self.a, self.b)

    def served(self):
        # no event when user does not click, therefore, pre-emptively add to b
        # if the ad is clicked, then this will be undone.
        self.b += 1

    def clicked(self):
        self.a += 1  # successful click
        self.b -= 1  # counter-act the 'no click' event added by served


def plotBandits(bandits, trial):
    x = np.linspace(0, 1, 200)  # axis between 0 and 1, 200 points.
    for bandito in bandits:
        # get probability of each value of x (likelihood of each win-rate)
        y = beta.pdf(x, bandito.a, bandito.b)
        plt.plot(x, y, label=bandito.name)
    plt.xlabel('Win Rate')
    plt.title('Bandit Distributions after %s trials' % trial)
    plt.legend()
    plt.show()


def get_ad():
    best = np.argmax([bandito.sample() for bandito in bandits])
    bandits[best].served()
    return best


def client(ad_A, ad_B):
    global bandits
    idxA = 0
    idxB = 0
    count = 0
    # quit when there's no data left for either ad
    while idxA < len(ad_A) and idxB < len(ad_B):
        best_idx = get_ad()
        if bandits[best_idx].name == 'A':
            action = ad_A[idxA]
            idxA += 1
        else:
            action = ad_B[idxB]
            idxB += 1

        if action == 1:
            # only click the ad if our dataset determines that we should
            # adds a click to the ad specified in the file 'click_data.csv'
            bandits[best_idx].clicked()

        # log some stats
        count += 1
        if count % 50 == 0:
            print("Seen %s ads, A: %s, B: %s" % (count, idxA, idxB))
            plotBandits(bandits, count)


if __name__ == '__main__':
    df = pd.read_csv('click_data.csv')
    ad_A = df[df['advertisement_id'] == 'A']
    ad_B = df[df['advertisement_id'] == 'B']
    ad_A = ad_A['action'].values
    ad_B = ad_B['action'].values

    # reminder of the click through rates of the complete dataset
    print("ad_A CTR (mean):", ad_A.mean())
    print("ad_B CTR (mean):", ad_B.mean())

    # initialize bandits
    bandits = [Bandit('A'), Bandit('B')]
    client(ad_A, ad_B)
