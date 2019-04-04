import numpy as np
from gridworld import standard_grid

convergence_threshold = 10e-4


class Policy(object):

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
        Printing out the values of each place on the grid according to a
        certain policy.
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
        Printing out the actions that will be taken at each
        place on the grid, according to the policy.
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


def policy_evaluation(policy, V, grid, gamma=.9):
    while True:
        biggest_change = 0
        for s in grid.actions.keys():
            old_v = V[s]  # keep track so we can measure change

            # action probabilites are deterministic in this policy
            grid.set_state(s)
            r = grid.move(policy[s])
            V[s] = r + gamma*V.get(the_grid.get_state(), 0)
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < convergence_threshold:
            break

    return V


def display_rewards(rewards, grid):
    """
    Printing out the values of each place on the grid according to a certain
    policy.
    """
    for i in range(grid.height):
        print("-" + "-------"*grid.width)
        for j in range(grid.width):
            if not j:
                print("|", end="")  # begin row with vertical line
            v = rewards.get((i, j), 0)
            if v >= 0:
                print(" %.2f |" % v, end="")
            else:
                print("%.2f |" % v, end="")  # -ve sign takes extra space
        print("")  # new line
    print("-" + "-------"*grid.width, end='\n\n')


if __name__ == '__main__':
    the_grid = standard_grid(step_cost=-0.1)
    possible_actions = list(the_grid.moves.keys())

    # print rewards ###### bake this in to the environment.
    print("Rewards:")
    display_rewards(the_grid.rewards, the_grid)

    poli = Policy(the_grid, gamma=.9)
    print("Randomly initialized policy:")
    poli.display(the_grid)

    # move policy evaluation, policy iteration, and value iteration into
    # Policy class. Maybe rename to DynamicPolicy or something, for
    # dynamic programming.
    # policy_changed = True
    # while policy_changed:
    #     V = policy_evaluation(policy, V, the_grid)
    #     policy_changed = False  # break out of loop if not changed
    #     for s, options in the_grid.actions.items():
    #         old_a = policy[s]  # keep track so we can measure change
    #         action_vals = []
    #         for a in options:
    #             # set position
    #             the_grid.set_state(s)
    #             # action probabilites are deterministic in this policy
    #             r = the_grid.move(a)
    #             action_vals.append(r + gamma*V.get(the_grid.get_state(), 0))
    #         policy[s] = options[np.argmax(action_vals)]
    #         policy_changed = True if policy[s] != old_a else policy_changed

    print("Learned values:")
    poli.display_values(the_grid)

    print("Learned policy:")
    poli.display(the_grid)
