import numpy as np
from gridworld import standard_grid

convergence_threshold = 10e-4


class DynamicPolicy(object):
    """
    Dynamic programming solution to the Markov Decision Process. This
    particular implementation has deterministic state transitions. See
    policy_iteration_windy.py for an example with probabilistic p(s',r|s,a).

    self.actions: state -> action policy
    self.V: values (expected returns) of each state according to this policy
    self.gamma: discount ratio of future expected returns
    """
    def __init__(self, grid, gamma=.9):
        # initialize state -> action policy
        self.actions = {}
        for s, options in grid.actions.items():
            self.actions[s] = np.random.choice(options)
        # initialize state-values
        self.V = {}
        for s in grid.actions.keys():
            self.V[s] = np.random.random()
        self.gamma = gamma

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
                if v >= 0:
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

    def policy_evaluation(self, grid):
        "Calculate values of each state according to the current policy."
        while True:
            biggest_change = 0
            for s in grid.actions.keys():
                old_v = self.V[s]  # keep track so we can measure change

                # action probabilites are deterministic in this policy
                grid.set_state(s)
                r = grid.move(self.actions[s])
                self.V[s] = r + self.gamma*self.V.get(grid.get_state(), 0)
                biggest_change = np.max(
                    [biggest_change, np.abs(old_v - self.V[s])])

            if biggest_change < convergence_threshold:
                break

    def update(self, grid):
        "Update state -> action policy based on state-value mappings."
        changed = False  # track whether policy has changed (for return)
        for s, options in grid.actions.items():
                old_a = self.actions[s]  # keep track so we can measure change
                action_vals = []
                for a in options:
                    # set position
                    grid.set_state(s)
                    # action probabilites are deterministic in this policy
                    r = grid.move(a)
                    action_vals.append(
                        r + self.gamma*self.V.get(grid.get_state(), 0))
                self.actions[s] = options[np.argmax(action_vals)]
                changed = True if self.actions[s] != old_a else changed
        return changed

    def policy_iteration(self, grid):
        """
        Calculate value of states based on current policy (policy evaluation),
        then update policy by greedily selecting the new best actions to
        maximize expected return.
        """
        changed = True
        while changed:
            self.policy_evaluation(grid)
            changed = self.update(grid)

    def value_iteration(self, grid):
        """
        Iterative maximization of state-value functions for each state, with
        evaluation of each possible action at every step. Then, once value
        functions have converged, update the state -> action policy.
        """
        while True:
            biggest_change = 0
            for s, options in grid.actions.items():
                old_v = self.V[s]  # keep track so we can measure change
                action_vals = []
                for a in options:
                    # set position
                    grid.set_state(s)
                    # action probabilites are deterministic in this policy
                    r = grid.move(a)
                    action_vals.append(
                        r + self.gamma*self.V.get(grid.get_state(), 0))
                self.V[s] = np.max(action_vals)
                biggest_change = np.max(
                    [biggest_change, np.abs(old_v - self.V[s])])

            if biggest_change < convergence_threshold:
                break
        # now update policy based on converged state-value functions
        self.update(grid)


if __name__ == '__main__':
    # generate a grid that penalizes each movement (encourage quick finish)
    the_grid = standard_grid(step_cost=-0.1)

    # print rewards associated with transitioning into each state on the grid
    print("Rewards:")
    the_grid.display_rewards()

    # randomly initialize a Dynamic Programming reinforcement learning policy
    policy = DynamicPolicy(the_grid, gamma=.9)
    print("Randomly initialized policy:")
    policy.display(the_grid)

    # learn the optimal policy / value function
    # policy.policy_iteration(the_grid)
    policy.value_iteration(the_grid)

    print("Learned values:")
    policy.display_values(the_grid)

    print("Learned policy:")
    policy.display(the_grid)
