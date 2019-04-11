import numpy as np
from gridworld import standard_grid

ALL_ACTIONS = ['U', 'D', 'L', 'R']


class MonteCarloApproxPolicy(object):
    """
    Reinforcement learning agent using Monte Carlo to learn the value function
    (evaluation) and dictate policy (improvement) for Gridworld.

    This version uses approximation methods, rather than tracking absolute
    values (expected returns) in dicts as monte_carlo.py does.
    """
    def __init__(self, grid, alpha=.001, gamma=.9, epsilon=.1):
        self.grid = grid  # environment
        self.actions = {}  # state -> action policy
        self.theta = np.random.randn(17)/np.sqrt(17)   # linear model params
        self.V = {}  # state-values (only calculated for display)
        # hyper-parameters
        self.alpha = alpha  # moving sample mean decay
        self.gamma = gamma  # discount ratio of future returns
        self.epsilon = epsilon  # chance to explore (epsilon-greedy)

    def display_values(self):
        """
        Printing out the values (expected returns) of each place on the grid
        according to this policy.
        """
        for i in range(self.grid.height):
            print("-" + "-------"*self.grid.width)
            for j in range(self.grid.width):
                if not j:
                    print("|", end="")  # begin row with vertical line
                v = self.V.get((i, j), 0)
                if np.isnan(v):
                    print("      |", end="")
                elif v >= 0:
                    print(" %.2f |" % v, end="")
                else:
                    print("%.2f |" % v, end="")  # -ve sign takes extra space
            print("")  # new line
        print("-" + "-------"*self.grid.width, end='\n\n')

    def display(self):
        """
        Printing out the actions that will be taken at each place on the grid,
        according to this policy.
        """
        for i in range(self.grid.height):
            print("-" + "-------"*self.grid.width)
            for j in range(self.grid.width):
                if not j:
                    print("|", end="")  # begin row with vertical line
                a = self.actions.get((i, j), ' ')
                print("   %s  |" % a, end="")
            print("")  # new line
        print("-" + "-------"*self.grid.width, end='\n\n')

    def play_episode(self):
        self.grid.reset()  # reset agent to start position
        history = []
        while not self.grid.gameover():
            s = self.grid.get_state()
            if np.random.random() > self.epsilon:
                # if no policy yet for current state, behave randomly
                a = self.actions.get(s, np.random.choice(ALL_ACTIONS))
            else:
                a = np.random.choice(ALL_ACTIONS)
            r = self.grid.move(a)  # take action and get reward
            history.append((s, a, r))

        return self.calculate_returns(history)

    def s2x(self, s):
        "Create state feature vector given s (coordinates of agent)."
        return np.array(
            [s[0]/self.grid.height, s[1]/self.grid.width,
             s[0]*s[1]/(self.grid.height+self.grid.width), 1])

    def sa2x(self, s, a):
        """
        Create state-action feature vector using state feature vector from
        self.s2x, multiplied by current action in one-hot style. Giving a
        feature vector of length len(s2x)*len(ALL_ACTIONS) + 1 bias term.
        """
        s_x = self.s2x(s)
        sa_x = np.concatenate([s_x*(a == act) for act in ALL_ACTIONS]+[[1]])
        return sa_x

    def calculate_returns(self, history):
        G = 0  # return of terminal state is zero
        s_a_returns = []  # returns of state-action pairs
        for s, a, r in reversed(history):
            G = r + self.gamma*G  # return for this state-action pair
            s_a_returns.append((s, a, G))

        return reversed(s_a_returns)  # forward chronological order

    def calculate_state_values(self):
        """
        Calculate state-value function from state-action value function.
        For each state s, V(s) = Q(s, pi(s)), where Q is approximated using a
        linear model with learned parameters (self.theta).
        """
        visited = set()
        for s in list(self.grid.actions.keys()):
            if s not in visited:
                visited.add(s)
                self.V[s] = self.theta @ self.sa2x(
                    s, self.actions.get(s, np.random.choice(ALL_ACTIONS)))

    def evaluation(self, N, first_visit=True):
        visited = set()
        for _ in range(N):
            s_a_returns = self.play_episode()
            for s, a, g in s_a_returns:
                if (s, a) not in visited or not first_visit:
                    visited.add((s, a))  # don't count again
                    x = self.sa2x(s, a)
                    self.theta = self.theta + self.alpha*(g - self.theta @ x)*x
            visited.clear()  # wipe for next episode

    def update(self):
        for s in list(self.grid.actions.keys()):
            options = [self.theta @ self.sa2x(s, a) for a in ALL_ACTIONS]
            self.actions[s] = ALL_ACTIONS[np.argmax(options)]

    def improve(self, N, first_visit=True):
        for _ in range(N):
            self.evaluation(1, first_visit=first_visit)
            self.update()  # alter policy with using new value approximations


if __name__ == '__main__':
    the_grid = standard_grid(step_cost=-0.1, windy=True)

    # print rewards associated with transitioning into each state on the grid
    print("Rewards:")
    the_grid.display_rewards()

    # create MC policy object. Action policy and Values not initialized
    policy = MonteCarloApproxPolicy(the_grid, alpha=.05, gamma=.9)

    # run until policy is unchanging (if windy, run a minimum number of times)
    iterations = policy.improve(10000, first_visit=True)
    policy.calculate_state_values()

    print("Learned values:")
    policy.display_values()

    print("Learned policy:")
    policy.display()
