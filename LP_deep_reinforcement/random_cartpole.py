import numpy as np
import matplotlib.pyplot as plt

import gym


def choose_action(state, params):
    "Only two actions in cart pole, left and right."
    return state.dot(params) > 0


def play_episode(env, params, max_steps=200):
    "Play round and return how many steps were taken before defeat."
    obsv = env.reset()  # start game and collect first observation
    for t in range(1, max_steps):
        obsv, reward, done, info = env.step(choose_action(obsv, params))
        if done:
            break
    return t


def random_search(env, epochs=100, trials=100):
    "Randomly sample weight parameters and keep the most succesful."
    best_t, top_times, params = 0, [], None
    for i in range(epochs):
        test_params = np.random.random(4)*2  # uniform -2 -> +2
        t = np.mean([play_episode(env, test_params) for _ in range(trials)])
        best_t, params = (t, test_params) if t > best_t else (best_t, params)
        top_times.append(best_t)

    return top_times, params


def main():
    # create cart pole environment, note: episodes stop at 200 steps.
    env = gym.make('CartPole-v0')

    # run random parameter search
    top_times, best_params = random_search(env)

    # display random search "progression" of survival time
    plt.plot(top_times)
    plt.xlabel('epochs')
    plt.ylabel('longest round')
    plt.show()

    # final test on best parameters that random search found.
    t = np.mean([play_episode(env, best_params) for _ in range(1000)])
    print('average time with best parameters:', t)


if __name__ == '__main__':
    main()
