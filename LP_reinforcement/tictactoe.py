import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

"""
Simple Reinforcement learning example using Tic-Tac-Toe, based on the lectures
so far in the Reinforcement Learning course (up until the basic code outline).
Have not looked at how LP implemented state-value mapping, but I imagine that
it is pretty similar to what I came up with using hashing.
"""


class TicTacToe(object):
    "Tic-Tac-Toe game for use as a reinforcement learning environment."
    def __init__(self):
        # 0=empty; -1=player1; 1=player2
        self.board = np.zeros((3, 3), dtype=np.int)
        self.gameover = False
        self.winner = 0  # -1=player1; 1=player2; 0=draw

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
    def __init__(self, number, alpha):
        self.number = number  # 1 (X) or 2 (O)
        self.alpha = alpha
        self.marker = int(-1) if number == 1 else int(1)
        self.values = {}
        self.game_history = []
        self.episodes = 0
        self.record = []

    def action(self, state):
        """
        Take in current board state and evaluate possible moves. Any states
        without a value are initialized to .5, the action that would lead to
        the highest value next state is chosen.
        """
        empty_spots = np.argwhere(state == 0)
        options = []
        for idx in empty_spots:
            placement = state.copy()
            placement[idx] = self.marker
            hsh = hash(placement.tobytes())
            if hsh not in self.values:
                self.values[hsh] = .5
            options.append(hsh)

        # epsilon-greedy, but with decaying epsilon.
        # Decided to try episodes/2 in denom to slow the decay.
        if np.random.random() > 1/(self.episodes/2+.000001):
            action_idx = np.argmax([self.values[opt] for opt in options])
        else:
            action_idx = np.random.choice(np.arange(len(options)))

        # also add current state to value dict
        hsh = hash(state.tobytes())
        if hsh not in self.values:
            self.values[hsh] = .5

        # pairs of state (s) and next state (s')
        self.game_history.append([hsh, options[action_idx]])

        return empty_spots[action_idx]  # index to place marker

    def update(self, reward):
        """
        Go back through each (s, s') pair and adjust their value based on
        the game result.
        """
        self.record.append(reward)
        last_value = reward
        for s, s_p in reversed(self.game_history):
            self.values[s_p] = self.values[s_p] + self.alpha*(
                                    last_value - self.values[s_p])
            self.values[s] = self.values[s] + self.alpha*(
                                    self.values[s_p] - self.values[s])
            last_value = self.values[s]

        self.game_history = []
        self.episodes += 1


def play_game(p1, p2, env, watch=False):
    play_order = shuffle([p1, p2])
    while(not env.gameover):
        for player in play_order:
            # player picks an action based on the state, and places a piece
            place = player.action(env.board)
            env.board[place[0], place[1]] = player.marker
            # print board state
            if watch:
                env.display()
            # check if game is over
            env.terminator()
            if env.gameover:
                break
    for player in play_order:
        # reward=0 for draws
        reward = 1 if env.winner == player.marker else 0
        player.update(reward)
    env.reset()  # clear board


if __name__ == '__main__':
    env = TicTacToe()
    player1 = Player(1, .01)
    player2 = Player(2, .01)
    for i in range(100000):
        play_game(player1, player2, env)

    for i in range(5):
        print('Test Game %d:' % i)
        play_game(player1, player2, env, watch=True)

    plt.plot(np.cumsum(player1.record)/np.arange(1, player1.episodes+1),
             label="player 1")
    plt.plot(np.cumsum(player2.record)/np.arange(1, player2.episodes+1),
             label="player 2")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.show()
