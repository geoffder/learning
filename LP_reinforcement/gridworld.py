# import numpy as np

"""
Similar to Lazy Programmer's, but not exactly. I'll see how much sense it
makes to bring it closer to his or diverge more as I begin implementing the
agents that will interact with this environment.
"""


class Grid(object):

    def __init__(self, dims, walls, terminals, start, step_cost=0):
        # basic layout and rules
        self.height, self.width = dims
        self.walls = set(walls)  # list of coordinate tuples converted to set
        self.moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        # states and actions
        self.terminals = set(terminals.keys())  # terminal positions
        self.rewards = terminals  # rewards initialized with terminal states
        self.step_cost = step_cost  # non-terminal spots are penalized
        self.actions = {}  # dict to be filled {(i, j): [actions]}
        self.enumerate_states()  # fill action and reward dicts
        # agent (start) position
        self.last_pos = start
        self.i, self.j = start

    def move(self, action):
        "Move Up, Down, Left, or Right (if legal, otherwise do nothing)"
        self.last_pos = (self.i, self.j)  # for undo-ing moves

        if action in self.actions.get(self.last_pos, []):
            self.i += self.moves[action][0]
            self.j += self.moves[action][1]

        return self.rewards.get((self.i, self.j), 0)  # return 0 if no entry

    def undo_move(self):
        self.i, self.j = self.last_pos

    def set_state(self, s):
        self.i, self.j = s

    def get_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        "Check if given state is terminal."
        return s in self.terminals

    def gameover(self):
        "Check if state agent is in is terminal (gameover, man.)"
        return (self.i, self.j) in self.terminals

    def enumerate_states(self):
        """
        Fill up rewards and action dicts for all legal states on the board.
        Walls will not have reward or action entries, as they can never be
        occupied, and terminal states will not have actions.
        """
        no_moves = self.terminals.union(self.walls)
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in no_moves:
                    self.rewards[(i, j)] = self.step_cost
                    self.actions[(i, j)] = self.possible_moves(i, j)

    def possible_moves(self, i, j):
        "Return list of legal moves from given grid position."
        legal = []
        for act in ['U', 'D', 'L', 'R']:
            new_i = i + self.moves[act][0]
            new_j = j + self.moves[act][1]
            if (new_i, new_j) not in self.walls:
                if new_i >= 0 and new_j >= 0:
                    if new_i < self.height and new_j < self.width:
                        legal.append(act)
        return legal

    def all_states(self):
        "Return set of all legal states on the board."
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_grid(step_cost=0):
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    terminals = {(0, 3): 1, (1, 3): -1}
    walls = [(1, 1)]
    the_grid = Grid((3, 4), walls, terminals, (2, 0), step_cost=step_cost)
    return the_grid


class Robot(object):

    def __init__(self):
        pass
