import numpy as np
from gridworld import standard_grid
from iterative_policy_eval import display_policy, display_values

convergence_threshold = 10e-4


def policy_evaluation(policy, V, grid, gamma=.9):
    while True:
        biggest_change = 0
        for s in grid.actions.keys():
            old_v = V[s]  # keep track so we can measure change
            new_v = 0  # accumulate value from all possibilities
            for a in all_actions:
                grid.set_state(s)
                # p(s',r|s,a) is not deterministic, despite π(a|s) being clear
                # (modelling uncertain state-transitions here)
                if a == policy[s]:
                    p_a = .5  # 50% chance of moving in intended direction
                else:
                    p_a = .5/3  # 16.6% chance of moving in other 3 directions
                r = grid.move(a)  # move and get associated reward
                new_v += p_a*(r + gamma*V.get(the_grid.get_state(), 0))
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            V[s] = new_v  # update

        if biggest_change < convergence_threshold:
            break

    return V


if __name__ == '__main__':
    # Agent will try to end game as quickly as possible with step costs this
    # high. Even if that means taking the negative terminal state.
    the_grid = standard_grid(step_cost=-1)
    all_actions = list(the_grid.moves.keys())

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
        for s in the_grid.actions.keys():
            old_a = policy[s]  # keep track so we can measure change
            action_vals = []
            for a in all_actions:  # intended action
                v = 0  # value of intended action, summed over possibilities
                for a2 in all_actions:  # actual action
                    # set position
                    the_grid.set_state(s)
                    # p(s',r|s,a) is not deterministic
                    if a == a2:
                        p_a = .5  # 50% chance of moving in intended direction
                    else:
                        p_a = .5/3  # 16.6% chance of moving in other 3 dirs
                    r = the_grid.move(a)
                    v += p_a*(r + gamma*V.get(the_grid.get_state(), 0))
                action_vals.append(v)
            policy[s] = all_actions[np.argmax(action_vals)]
            policy_changed = True if policy[s] != old_a else policy_changed

    print("Learned values:")
    display_values(V, the_grid)
    print("")

    print("Learned policy:")
    display_policy(policy, the_grid)
    print("")
