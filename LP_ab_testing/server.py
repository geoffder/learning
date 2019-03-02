import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import beta

# create an app
app = Flask(__name__)


# define bandits
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


# initialize bandits
bandits = [Bandit('A'), Bandit('B')]


@app.route('/get_ad')
def get_ad():
    best = np.argmax([bandito.sample() for bandito in bandits])
    bandits[best].served()
    return jsonify({'advertisement_id': bandits[best].name})


@app.route('/click_ad', methods=['POST'])
def click_ad():
    result = 'OK'
    if request.form['advertisement_id'] == 'A':
        bandits[0].clicked()
        pass
    elif request.form['advertisement_id'] == 'B':
        bandits[1].clicked()
        pass
    else:
        result = 'Invalid Input.'

    # nothing to return really
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8888')
