import numpy as np
from gridworld import standard_grid
from iterative_policy_eval import display_policy, display_values

convergence_threshold = 10e-4


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


if __name__ == '__main__':
    the_grid = standard_grid(step_cost=-0.1)
    possible_actions = list(the_grid.moves.keys())

    # print rewards
    print("Rewards:")
    display_values(the_grid.rewards, the_grid)
    print("")

    # state -> action
    # randomly choose an action and update as we learn
    policy = {}
    for s, options in the_grid.actions.items():
        policy[s] = np.random.choice(options)
    print("Randomly initialized policy:")
    display_policy(policy, the_grid)

    V = {}
    for s in the_grid.actions.keys():
        V[s] = np.random.random()
    gamma = .9

    policy_changed = True
    while policy_changed:
        V = policy_evaluation(policy, V, the_grid)
        policy_changed = False  # break out of loop if not changed
        for s, options in the_grid.actions.items():
            old_a = policy[s]  # keep track so we can measure change
            action_vals = []
            for a in options:
                # set position
                the_grid.set_state(s)
                # action probabilites are deterministic in this policy
                r = the_grid.move(a)
                action_vals.append(r + gamma*V.get(the_grid.get_state(), 0))
            policy[s] = options[np.argmax(action_vals)]
            policy_changed = True if policy[s] != old_a else policy_changed

    print("Learned values:")
    display_values(V, the_grid)
    print("")

    print("Learned policy:")
    display_policy(policy, the_grid)
    print("")
