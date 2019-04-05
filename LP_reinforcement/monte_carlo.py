import numpy as np
from gridworld import standard_grid

ALL_ACTIONS = ['U', 'D', 'L', 'R']


class MonteCarloPolicy(object):

    def __init__(self, alpha=.9, gamma=.9, epsilon=.1, sample_mean=False):
        # state -> action policy, instead of initializing, I'll take a
        # discovering states as I go approach
        self.actions = {}
        self.V = {}  # state-values (add states as game is played)
        self.Q = {}  # values of state-action pairs
        self.sample_mean = sample_mean  # use sample mean vs moving average
        self.N = {}  # number of samples of state-action pairs
        self.alpha = alpha  # moving sample mean decay
        self.gamma = gamma  # discount ratio of future returns
        self.epsilon = epsilon  # chance to explore (epsilon-greedy)

    def display_values(self, grid):
        """
        Printing out the values (expected returns) of each place on the grid
        according to this policy.
        """
        for i in range(grid.height):
            print("-" + "-------"*grid.width)
            for j in range(grid.width):
                if not j:
                    print("|", end="")  # begin row with vertical line
                v = self.V.get((i, j), 0)
                if np.isnan(v):
                    print("     |", end="")
                elif v >= 0:
                    print(" %.2f |" % v, end="")
                else:
                    print("%.2f |" % v, end="")  # -ve sign takes extra space
            print("")  # new line
        print("-" + "-------"*grid.width, end='\n\n')

    def display(self, grid):
        """
        Printing out the actions that will be taken at each place on the grid,
        according to this policy.
        """
        for i in range(grid.height):
            print("-" + "-------"*grid.width)
            for j in range(grid.width):
                if not j:
                    print("|", end="")  # begin row with vertical line
                a = self.actions.get((i, j), ' ')
                print("   %s  |" % a, end="")
            print("")  # new line
        print("-" + "-------"*grid.width, end='\n\n')

    def play_episode(self, grid):
        grid.reset()  # reset agent to start position
        history = []
        while not grid.gameover():
            s = grid.get_state()
            if np.random.random() > self.epsilon:
                # if no policy yet for current state, behave randomly
                a = self.actions.get(s, np.random.choice(ALL_ACTIONS))
            else:
                a = np.random.choice(ALL_ACTIONS)
            r = grid.move(a)  # take action and get reward
            history.append((s, a, r))

        return self.calculate_returns(history)

    def calculate_returns(self, history):
        G = 0  # return of terminal state is zero
        s_a_returns = []  # returns of state-action pairs
        for s, a, r in reversed(history):
            G = r + self.gamma*G  # return for this state-action pair
            s_a_returns.append((s, a, G))

        return reversed(s_a_returns)  # forward chronological order

    def calculate_state_values(self):
        visited = set()
        for (s, _) in self.Q.keys():
            if s not in visited:
                visited.add(s)
                self.V[s] = np.max(
                    [self.Q.get((s, a), np.NaN) for a in ALL_ACTIONS])

    def evaluation(self, grid, N, first_visit=True):
        visited = set()
        for _ in range(N):
            s_a_returns = self.play_episode(grid)
            for s, a, g in s_a_returns:
                if (s, a) not in visited or not first_visit:
                    visited.add((s, a))  # don't count again
                    old_q = self.Q.get((s, a), np.random.random())
                    if self.sample_mean:
                        # use sample mean to estimate Q
                        self.N[(s, a)] = self.N.get((s, a), 0) + 1
                        self.Q[(s, a)] = (
                            1 - 1/self.N[(s, a)])*old_q + g/self.N[(s, a)]
                    else:
                        # use exponential moving average to estimate Q
                        self.Q[(s, a)] = (1 - self.alpha)*old_q + self.alpha*g
                        # or equivalently
                        # self.Q[(s, a)] = old_q + self.alpha*(g - old_q)
            visited.clear()  # wipe for next episode

    def update(self):
        changed = False  # track whether policy has changed (for return)
        visited = set()
        for (s, _) in self.Q.keys():
            if s not in visited:
                visited.add(s)
                old_a = self.actions.get(s, 0)  # keep track (check for change)
                options = [self.Q.get((s, a), np.NaN) for a in ALL_ACTIONS]
                self.actions[s] = ALL_ACTIONS[np.argmax(options)]
                changed = True if self.actions[s] != old_a else changed
        return changed

    def improve(self, grid, first_visit=True):
        iterations = 0
        changed = True
        while changed or (iterations < 1000 and grid.windy):
            self.evaluation(grid, 1, first_visit=first_visit)
            changed = self.update()
            iterations += 1
        return iterations


if __name__ == '__main__':
    the_grid = standard_grid(step_cost=-0.1, windy=True)

    # print rewards associated with transitioning into each state on the grid
    print("Rewards:")
    the_grid.display_rewards()

    # create MC policy object. Action policy and Values not initialized
    policy = MonteCarloPolicy(alpha=.9, gamma=.9, sample_mean=True)

    # run until policy is unchanging (if windy, run a minimum number of times)
    iterations = policy.improve(the_grid, first_visit=True)
    policy.calculate_state_values()  # calculate values of each state from Q

    print("Learned values (after %d iterations):" % iterations)
    policy.display_values(the_grid)

    print("Learned policy (after %d iterations):" % iterations)
    policy.display(the_grid)
