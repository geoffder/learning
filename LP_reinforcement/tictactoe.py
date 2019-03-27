import numpy as np
import matplotlib.pyplot as plt


class TicTacToe(object):

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
        elif np.count_nonzero == 9:
            self.gameover = True

    def reset(self):
        self.board *= 0
        self.gameover = 0
        self.winner = 0

    def display(self):
        lookup = {-1: 'X', 0: '_', 1: 'O'}
        XOboard = np.array(
            [[lookup[ele] for ele in row] for row in self.board])
        print(XOboard)


class Player(object):

    def __init__(self, number, alpha):
        self.number = number  # 1 (X) or 2 (O)
        self.alpha = alpha
        self.marker = -1 if number == 1 else 1
        self.values = {}
        self.state_idx = 0
        self.game_history = []
        self.episodes = 0

    def action(self, state):
        empty_spots = np.argwhere(state == 0)
        options = []
        for idx in empty_spots:
            placement = state.copy()
            placement[idx] = self.marker
            hsh = hash(bytes(placement))
            if hsh not in self.values:
                self.values[hsh] = .5
            options.append(hsh)

        # epsilon-greedy, but with decaying epsilon
        if np.random.random() > 1/(self.episodes+.000001):
            action_idx = np.argmax([self.values[opt] for opt in options])
        else:
            action_idx = np.random.choice(np.arange(len(options)))

        # pairs of state (s) and next state (s')
        self.game_history.append([hash(bytes(state)), options[action_idx]])

        return empty_spots[action_idx]  # index to place marker

    def update(self, reward):
        """
        Go back through each (s, s') pair and adjust their value based on
        the game result.
        """
        pass


if __name__ == '__main__':
    env = TicTacToe()
    env.board = np.array([[-1, 0, 1], [1, 0, 1], [-1, 0, -1]])
    env.display()
    env.terminator()
    print(env.gameover, env.winner)
