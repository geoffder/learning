import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

"""
Need to change the way this whole thing works in order to implement as TD
learning. Need to use the action coordinates for Q. Also, how to properly do
SARSA? Pretend that an agent can move twice in a row? Seems wrong... Read up.
"""


class TicTacToe(object):
    "Tic-Tac-Toe game for use as a reinforcement learning environment."
    def __init__(self):
        # 0=empty; -1=player1; 1=player2
        self.board = np.zeros((3, 3), dtype=np.int)
        self.gameover = False
        self.winner = 0  # -1=player1; 1=player2; 0=draw
        self.win = 1
        self.lose = -1
        self.tie = -.5
        self.t = 1  # count number of episodes

    def terminator(self):
        "Check for end of game / terminal state"
        # sums of all lines on the board, a victory will result in one or more
        # lines summing to -3 or +3, for player1 and player2 respectively
        allLines = np.concatenate([
            self.board.sum(axis=0), self.board.sum(axis=1),
            [np.diag(self.board).sum()],
            [np.diag(np.flip(self.board, axis=0)).sum()]
        ])
        if -3 in allLines:
            self.gameover = True
            self.winner = -1
        elif 3 in allLines:
            self.gameover = True
            self.winner = 1
        elif np.count_nonzero(self.board) > 8:
            self.gameover = True

    def get_reward(self, marker):
        if self.gameover and self.winner == marker:
            return self.win
        else:
            return 0  # any move that doesn't win

    def reset(self):
        "Wipe board."
        self.board *= 0
        self.gameover = False
        self.winner = 0

    def display(self):
        "Print board state to terminal."
        lookup = {-1: 'X', 0: '_', 1: 'O'}
        XOboard = np.array(
            [[lookup[ele] for ele in row] for row in self.board])
        print(XOboard, end='\n\n')


class Player(object):
    "Reinforcement learning agent."
    def __init__(self, number, policy=None):
        self.number = number  # 1 (X) or 2 (O)
        self.marker = int(-1) if number == 1 else int(1)
        self.policy = Dumb() if policy is None else policy
        self.policy.marker = self.marker
        self.last_s, self.last_a = None, None
        self.record = []
        self.greed = False

    def action(self):
        """
        Take in current board state perform actions according to policy.
        """
        state = env.board
        hsh = hash(state.tobytes())
        options = [tuple(loc) for loc in np.argwhere(state == 0)]
        if self.greed:
            self.policy.greedy(hsh, options)
        else:
            self.last_a = self.policy.make_move(
                self.last_s, self.last_a, self.last_options, hsh, options)
            self.last_s = hsh
            self.last_options = options

    def defeat(self):
        hsh = hash(env.board.tobytes())
        self.policy.loser(self.last_s, self.last_a, self.last_options, hsh)

    def new_game(self):
        self.last_s, self.last_a, self.last_options = None, None, None


class Human(object):

    def __init__(self, number):
        self.number = number  # 1 (X) or 2 (O)
        self.marker = int(-1) if number == 1 else int(1)

    def action(self):
        # show human the board
        env.display()
        # get action (prompt)
        while True:
            user_in = input("Input coordinates delimited by comma: ")
            coords = tuple([int(ele) for ele in user_in.split(',')])
            state = env.board
            options = [tuple(loc) for loc in np.argwhere(state == 0)]
            if coords in options:
                env.board[(coords)] = self.marker
                env.terminator()
                break
            else:
                print("Try again...")

    def new_game(self):
        pass


class TemporalDifferencePolicy(object):
    """
    Reinforcement learning agent using Temporal Difference Learning to learn
    the value function and dictate policy for Gridworld.
    """
    def __init__(self, alpha=.1, gamma=.9, epsilon=.1, eps_decay=True):
        # state -> action policy, instead of initializing, I'll take a
        # discovering states as I go approach
        self.actions = {}
        self.V = {}  # state-values (add states as game is played)
        self.Q = {}  # values of state-action pairs
        self.NSA = {}  # number of visits to state-action pairs
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount ratio of future returns
        self.epsilon = epsilon  # chance to explore (epsilon-greedy)
        self.eps_decay = eps_decay
        self.marker = 0

    def make_move(self, last_s, last_a, last_options, s1, options):
        # choose action
        a1 = self.epsilon_greedy(s1, options, env.t)
        # execute
        env.board[a1] = self.marker
        env.terminator()
        r = env.get_reward(self.marker)  # check for win
        s2 = hash(env.board.tobytes())
        if last_s is not None:
            # reward is 0 if the game wasn't over, which it wasn't
            self.TD0(last_s, last_a, -.1, s1, a1, last_options)
        if env.gameover:
            # update now since there won't be a next move
            a2 = None  # no action from terminal state
            r = env.tie if r < env.win else r
            self.TD0(s1, a1, r, s2, a2, options)
        return a1

    def loser(self, last_s, last_a, last_options, terminus):
        # negative reward for game ending on opponents turn
        r = env.tie if env.winner == 0 else env.lose
        self.TD0(last_s, last_a, r, terminus, None, last_options)

    def TD0(self, s1, a1, r, s2, a2, options):
        """
        TD0 prediction with SARSA control (greedy policy improvement). Also
        can function as Q-Learning without SARSA, as it is agnostic to the
        selection of a1 and a2, whether on-policy or off-policy.
        Learning rate decays for each state independently as a function of the
        number of visits (and therefore, learning events) it has had.
        """
        # initialize values that don't exist
        if a2 is None:
            self.Q[(s2, a2)] = 0  # terminal states have no value
        if (s2, a2) not in self.Q:
            # should this be initialized 0 since it may never be visited?
            self.Q[(s2, a2)] = 0  # np.random.random()  # initialize

        # increment visit counter for (s1, a1) and calculate learning rate
        self.NSA[(s1, a1)] = self.NSA.get((s1, a1), 0) + 1
        lr = self.alpha/(1 + self.NSA[(s1, a1)]*.005)  # decaying

        # update state-action value
        # old_q = self.Q.get((s1, a1), np.random.random())
        old_q = self.Q.get((s1, a1), 0)  # try 0 initializing
        # forgot to change this before, must take best value of Q for the
        # next state for Q learning (even if not the action performed)!
        next_q = np.max([self.Q.get((s2, act), 0) for act in options])
        self.Q[(s1, a1)] = old_q + lr*(r + self.gamma*next_q - old_q)
        # self.Q[(s1, a1)] = old_q + lr*(
        #     r + self.gamma*self.Q[(s2, a2)] - old_q)

        # update policy
        values = [self.Q.get((s1, act), 0) for act in options]
        self.actions[s1] = options[np.argmax(values)]

    def epsilon_greedy(self, s, options, t):
        """
        Epsilon-greedy action selection given a state s. 'Time' t is used to
        calculate decaying epsilon, if activated for this policy.
        """
        eps = 1/t/100 if self.eps_decay else self.epsilon
        if np.random.random() > eps:
            # if no policy yet for current state, behave randomly
            a = self.actions.get(s, options[np.random.randint(len(options))])
        else:
            a = options[np.random.randint(len(options))]
        return a

    def greedy(self, s, options):
        "Greedy action selection given a state s. Return highest value action."
        a = self.actions.get(s, options[np.random.randint(len(options))])
        env.board[a] = self.marker
        env.terminator()
        return a


class Dumb(object):

    def __init__(self):
        self.marker = 0

    def make_move(self, last_s, last_a, last_options, s, options):
        # a = np.random.choice(options)
        a = options[np.random.randint(len(options))]
        env.board[a] = self.marker
        env.terminator()

    def loser(self, last_s, last_a, last_options, terminus):
        pass


def play_game(p1, p2, watch=False):
    play_order = shuffle([p1, p2])
    for player in play_order:
        player.new_game()
    while(not env.gameover):
        for i, player in enumerate(play_order):
            # player picks an action based on the state, and places a piece
            player.action()
            if watch:
                env.display()  # print board state
            if env.gameover:
                break
    for j, player in enumerate(play_order):
        if player.marker != env.winner:
            if j != i:
                # give player final update if they didn't go last
                player.defeat() if not player.greed else 0
            reward = 0
        else:
            reward = 1
        player.record.append(reward)
    env.t += 1
    env.reset()  # clear board


def wargames(human, machine):
    print("Beginning War Against the Machine...")
    mark = 'X' if human.marker == -1 else 'O'
    print("Human, you are '%s'" % mark)
    machine.greed = True
    while True:
        print("\nFight!")
        play_order = shuffle([human, machine])
        for player in play_order:
            player.new_game()
        while(not env.gameover):
            for i, player in enumerate(play_order):
                player.action()  # board printed and input prompted for human
                if env.gameover:
                    break
        env.display()
        if env.winner == human.marker:
            result = "You win!"
        elif env.winner == machine.marker:
            result = "You lose!"
        else:
            result = "Stalemate!"
        print("Game Over! Man. %s" % result, end="\n\n")
        env.reset()  # clear board

        again = input("Play again? y or n: \n")
        if again.lower() == 'n':
            break


if __name__ == '__main__':
    env = TicTacToe()  # this is global
    SARSA = False
    if SARSA:
        # plays on-policy. Seems like agent pigeon-holes itself quite hard,
        # while playing on-policy. Doesn't learn accurate values over a
        # diversity of states.
        player1 = Player(
            1, policy=TemporalDifferencePolicy(alpha=.05, gamma=.9))
    else:
        # plays off-policy (Q-learning)
        player1 = Player(
            1, policy=TemporalDifferencePolicy(
                alpha=.05, gamma=.9, eps_decay=False, epsilon=1))

    # plays off-policy (Q-learning)
    player2 = Player(
        2, policy=TemporalDifferencePolicy(
            alpha=.05, gamma=.9, eps_decay=False, epsilon=1))

    print("p1 = X; p2 = O")

    for i in range(100000):
        play_game(player1, player2)

    # put both players on-policy (fully greedy) and observe games
    player1.greed = True
    player2.greed = True
    for i in range(5):
        print('Trained vs Trained Game %d:' % i)
        play_game(player1, player2, watch=True)

    plt.plot(np.cumsum(player1.record)/np.arange(1, env.t),
             label="player 1")
    plt.plot(np.cumsum(player2.record)/np.arange(1, env.t),
             label="player 2")
    plt.title("Trained Agent vs Trained Agent")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.show()

    # test trained agent against a random one
    player1.record = []
    player2 = Player(2)  # Reset player2 to Dumb with no record
    beatdowns = 1000
    for i in range(beatdowns):
        play_game(player1, player2)

    plt.plot(np.cumsum(player1.record)/np.arange(1, beatdowns+1),
             label="player 1")
    plt.plot(np.cumsum(player2.record)/np.arange(1, beatdowns+1),
             label="player 2")
    plt.title("Trained Agent vs Dumb Agent")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.show()

    # for i in range(5):
    #     print('Trained vs Dumb Game %d:' % i)
    #     play_game(player1, player2, watch=True)

    guy = Human(2)
    wargames(guy, player1)
