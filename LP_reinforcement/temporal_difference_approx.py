import numpy as np
from gridworld import standard_grid

ALL_ACTIONS = ['U', 'D', 'L', 'R']


class TemporalDifferenceApproxPolicy(object):
    """
    Reinforcement learning agent using Temporal Difference Learning to learn
    the value function and dictate policy for Gridworld.
    """
    def __init__(self, grid, alpha=.1, gamma=.9, epsilon=.1, eps_decay=True):
        self.grid = grid  # environment
        self.actions = {}  # state -> action policy
        self.theta = np.random.randn(17)/np.sqrt(17)  # linear model params
        self.V = {}  # state-values (only calculated for display)
        self.NSA = {}  # number of visits to state-action pairs
        # hyper-parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount ratio of future returns
        self.epsilon = epsilon  # chance to explore (epsilon-greedy)
        self.eps_decay = eps_decay  # whether to decay epsilon over time

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

    def Qlearning(self, T, SARSA=True):
        "Run through T episodes of play, using TD0 SARSA policy improvement."
        for t in range(1, T+1):
            self.play_episode(t, SARSA)

    def play_episode(self, t, SARSA):
        "Run an episode with TD0 SARSA value and policy updates at every step."
        self.grid.reset()  # reset agent to start position
        s1 = self.grid.get_state()  # initial state
        a1 = self.epsilon_greedy(s1, t)
        while True:
            if not SARSA:
                # don't necessarily use a2 from previous step, could use a
                # completely random action for Q-learning.
                # Set eps_decay=False with epsilon=1 for off-policy Q-learning
                a1 = self.epsilon_greedy(s1, t)
            r = self.grid.move(a1)  # take action and get reward
            s2 = self.grid.get_state()
            isTerminal = self.grid.gameover()
            if not isTerminal:
                a2 = self.epsilon_greedy(s2, t) if SARSA else self.greedy(s2)
            else:
                a2 = None
            self.TD0_SARSA(s1, a1, r, s2, a2, SARSA)
            s1, a1 = s2, a2
            if isTerminal:
                break

    def TD0_SARSA(self, s1, a1, r, s2, a2, SARSA):
        """
        TD0 prediction with SARSA control (greedy policy improvement). Also
        can function as Q-Learning without SARSA, as it is agnostic to the
        selection of a1 and a2, whether on-policy or off-policy.
        Learning rate decays for each state independently as a function of the
        number of visits (and therefore, learning events) it has had.
        """

        # increment visit counter for (s1, a1) and calculate learning rate
        self.NSA[(s1, a1)] = self.NSA.get((s1, a1), 0) + 1
        lr = self.alpha/(1 + self.NSA[(s1, a1)]*.005)  # decaying

        # update state-action approximation parameters
        x1 = self.sa2x(s1, a1)
        q1 = self.theta @ x1
        if SARSA:
            q2 = self.theta @ self.sa2x(s2, a2)
        else:
            # must take best value of Q for the next state for Q learning
            # (even if not the action performed!)
            q2 = np.max([self.theta @ self.sa2x(s2, act)
                         for act in ALL_ACTIONS])
        self.theta = self.theta + lr*(r + self.gamma*q2 - q1)*x1

        # update policy
        options = [self.theta @ self.sa2x(s1, act) for act in ALL_ACTIONS]
        self.actions[s1] = ALL_ACTIONS[np.argmax(options)]

    def epsilon_greedy(self, s, t):
        """
        Epsilon-greedy action selection given a state s. 'Time' t is used to
        calculate decaying epsilon, if activated for this policy.
        """
        eps = 1/t/100 if self.eps_decay else self.epsilon
        if np.random.random() > eps:
            # if no policy yet for current state, behave randomly
            a = self.actions.get(s, np.random.choice(ALL_ACTIONS))
        else:
            a = np.random.choice(ALL_ACTIONS)
        return a

    def greedy(self, s):
        "Greedy action selection given a state s. Return highest value action."
        return self.actions.get(s, np.random.choice(ALL_ACTIONS))

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


if __name__ == '__main__':
    the_grid = standard_grid(step_cost=-0.1, windy=True)

    # print rewards associated with transitioning into each state on the grid
    print("Rewards:")
    the_grid.display_rewards()

    # Learn using SARSA or off-policy Q-learning control strategy.
    # Both have very similar results (optimal value-function and policy found),
    # though off-policy Q-learning takes longer as the Agent is not actively
    # trying to make it to the goal.
    SARSA = False
    if SARSA:
        # SARSA follows an explore-exploit strategy, so agent moves through
        # environment semi-greedily (epsilon-greedy here) according to policy.
        policy = TemporalDifferenceApproxPolicy(the_grid, alpha=.05, gamma=.9)
        policy.Qlearning(10000, SARSA=SARSA)
    else:
        # In off-policy Q-learning, the agent is not following any policy and
        # instead behaves randomly, despite continually updating Q towards the
        # optimal value-function Q*.
        policy = TemporalDifferenceApproxPolicy(
            the_grid, alpha=.05, gamma=.9, eps_decay=False, epsilon=1)
        policy.Qlearning(10000, SARSA=SARSA)

    policy.calculate_state_values()  # calculate values of each state from Q
    print("Learned values:")
    policy.display_values()

    print("Learned policy:")
    policy.display()
