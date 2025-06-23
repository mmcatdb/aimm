import numpy as np
from Game import Game
from State import State

class RandomAgent():
    def __init__(self, game: Game):
        self.game = game

    def play(self, state: State):
        action = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(state)
        while valids[action] != 1:
            action = np.random.randint(self.game.getActionSize())

        return action

class HumanAgent():
    def __init__(self, game: Game):
        self.game = game

    def play(self, state: State):
        valid = self.game.getValidMoves(state)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i / self.game.n), int(i % self.game.n), end = "] ")

        while True:
            inputMove = input()
            inputAction = inputMove.split(" ")
            if len(inputAction) == 2:
                try:
                    x, y = [int(i) for i in inputAction]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')

        return a

class GreedyAgent():
    def __init__(self, game: Game):
        self.game = game

    def play(self, state: State):
        valids = self.game.getValidMoves(state)
        candidates = []
        for action in range(self.game.getActionSize()):
            if valids[action] == 0:
                continue
            nextState = self.game.getNextState(state, action)
            score = self.game.getScore(nextState)
            candidates += [(-score, action)]
        candidates.sort()

        return candidates[0][1]
