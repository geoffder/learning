import numpy as np
from gridworld import standard_grid

convergence_threshold = 10e-4


def display_values(values, grid):
    """
    Printing out the values of each place on the grid according to a certain
    policy.
    """
    for i in range(grid.height):
        print("-" + "-------"*grid.width)
        for j in range(grid.width):
            if not j:
                print("|", end="")  # begin row with vertical line
            v = values.get((i, j), 0)
            if v >= 0:
                print(" %.2f |" % v, end="")
            else:
                print("%.2f |" % v, end="")  # -ve sign takes extra space
        print("")  # new line
    print("-" + "-------"*grid.width)


def display_policy(policy, grid):
    """
    Printing out the actions that will be taken at each
    place on the grid, according to the policy.
    """
    for i in range(grid.height):
        print("-" + "-------"*grid.width)
        for j in range(grid.width):
            if not j:
                print("|", end="")  # begin row with vertical line
            a = policy.get((i, j), ' ')
            print("   %s  |" % a, end="")
        print("")  # new line
    print("-" + "-------"*grid.width)


if __name__ == '__main__':
    the_grid = standard_grid()
    states = the_grid.all_states()

    # # # poilcy with uniformly random actions # # #
    V = {}
    for s in states:
        V[s] = 0  # initialize all state values to 0
    gamma = 1  # discount factor

    # repeat until convergence
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]  # keep track so we can measure change

            # terminal states have no value (no future returns)
            if s in the_grid.actions:
                new_v = 0  # accumulate the value of possibilities
                # each action has equal probability in this policy
                p_a = 1 / len(the_grid.actions[s])
                for a in the_grid.actions[s]:
                    the_grid.set_state(s)
                    r = the_grid.move(a)
                    new_v += p_a * (r + gamma*V[the_grid.get_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < convergence_threshold:
            break

    print("Values for uniformly random actions:")
    display_values(V, the_grid)
    print("")

    # # # fixed policy # # #
    # note: this is designed for the standard grid
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    print("The fixed policy:")
    display_policy(policy, the_grid)
    print("")

    V = {}
    for s in states:
        V[s] = 0  # initialize all state values to 0
    # observe how value of states decreases with distance from reward
    gamma = .9  # discount factor

    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]  # keep track so we can measure change

            # terminal states have no value (no future returns)
            if s in the_grid.actions:
                new_v = 0  # accumulate the value of possibilities
                # unnecessary loop, but leaving in for comparison
                for a in the_grid.actions[s]:
                    # action probabilites are deterministic in this policy
                    p_a = 1 if a == policy[s] else 0
                    the_grid.set_state(s)
                    r = the_grid.move(a)
                    new_v += p_a * (r + gamma*V[the_grid.get_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < convergence_threshold:
            break

    print("Values for fixed actions:")
    display_values(V, the_grid)
    print("")
