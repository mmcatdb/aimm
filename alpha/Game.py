import numpy as np
from numpy.typing import NDArray
from IGame import IGame
from State import State

class Game(IGame[State]):
    def __init__(self, n):
        self.n = n

    def getStateSize(self) -> tuple[int, int]:
        # (a, b) tuple
        return (self.n, self.n)

    def getActionSize(self) -> int:
        # return number of actions
        return self.n * self.n + 1
    
    def getInitState(self) -> State:
        return State(self.n)

    def getNextState(self, state: State, action: int) -> State:
        # if agent takes action on state, return next state
        # action must be a valid move
        if action == self.n * self.n:
            return state
        
        newState = State.copy(state)
        move = (int(action / self.n), action % self.n)
        newState.executeMove(move)

        return newState

    def getValidMoves(self, state: State) -> NDArray[np.int32]:
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()

        legalMoves = state.getLegalMoves()
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        
        for x, y in legalMoves:
            valids[self.n * x + y] = 1

        return np.array(valids)

    def getGameEnded(self, state: State) -> float:
        # TODO
        if isTerminal(state):  # e.g., max depth, or no valid moves
            reward = evaluatState(state)  # your framework’s score
            cost = getTotalActionCost(state)
            finalScore = reward - cost
            return finalScore
        
        return 0

        # b = State(self.n)
        # b.pieces = np.copy(state)
        # if b.hasLegalMoves():
        #     return 0
        # if b.countDiff() > 0:
        #     return 1
        # return -1

    def getScore(self, state: State) -> int:
        return state.countDiff()
    
    def getStringRepresentation(self, state: State) -> str:
        return state.getStringRepresentation()

    @staticmethod
    def display(state: State) -> None:
        n = state.pieces.shape[0]
        print("   ", end = "")
        for y in range(n):
            print(y, end = " ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end = "")    # print the row #
            for x in range(n):
                piece = state[y][x]    # get the piece to print
                print(Game.squareContent[piece], end = " ")
            print("|")

        print("-----------------------")

    squareContent = {
        -1: "X",
        +0: "-",
        +1: "O"
    }
