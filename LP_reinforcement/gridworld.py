# import numpy as np

"""
Similar to Lazy Programmer's, but not exactly. I'll see how much sense it
makes to bring it closer to his or diverge more as I begin implementing the
agents that will interact with this environment.
"""


class Grid(object):

    def __init__(self, dims, walls, terminals, start, step_cost=0):
        # basic layout
        self.width, self.height = dims
        self.walls = set(walls)  # list of coordinate tuples converted to set
        # states and actions
        self.terminals = set(terminals.keys())  # terminal positions
        self.rewards = terminals  # rewards initialized with terminal states
        self.step_cost = step_cost  # non-terminal spots are penalized
        self.actions = {}  # dict to be filled {(i, j): [actions]}
        # agent start position
        self.last_pos = start
        self.i, self.j = start

    def move(self, action):
        "Move Up, Down, Right, or Left (if legal)"
        self.last_pos = (self.i, self.j)
        if action == "U" and self.i-1 > 0:
            self.i -= 1
        elif action == "D" and self.i+1 < self.height:
            self.i += 1
        elif action == "R" and self.j+1 < self.width:
            self.j += 1
        elif action == "L" and self.j-1 > 0:
            self.j -= 1

        # bounce back if moved in to a wall
        if (self.i, self.j) in self.walls:
            self.i, self.j = self.last_pos

        return self.rewards.get((self.i, self.j), 0)  # return 0 if nothing

    def undo_move(self):
        self.i, self.j = self.last_pos

    def is_terminal(self, s):
        "Check if given state is terminal."
        return s in self.terminals

    def gameover(self):
        "Check if state agent is in is terminal (gameover, man.)"
        return (self.i, self.j) in self.terminals

    def all_states(self):
        """
        Not sure if I will be using this as LP does yet, including so I
        remember in case I'm not sure how I want to implement.
        """
        return set(self.actions.keys() + self.rewards.keys())


def standard_grid():
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    terminals = {(0, 3): 1, (1, 3): -1}
    walls = [(1, 1)]
    the_grid = Grid((3, 4), walls, terminals, (2, 0), neg_mode=False)
    return the_grid
