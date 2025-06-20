from Game import Game
from othello.OthelloBoard import OthelloBoard
import numpy as np
from numpy.typing import NDArray

class OthelloGame(Game[OthelloBoard]):
    def __init__(self, n):
        self.n = n

    def getBoardSize(self) -> tuple[int, int]:
        # (a, b) tuple
        return (self.n, self.n)

    def getActionSize(self) -> int:
        # return number of actions
        return self.n * self.n + 1
    
    def getInitState(self) -> OthelloBoard:
        return OthelloBoard(self.n)

    def getNextState(self, board: OthelloBoard, action: int) -> OthelloBoard:
        # if player takes action on board, return next board
        # action must be a valid move
        if action == self.n * self.n:
            return board
        
        newBoard = OthelloBoard.copy(board)
        move = (int(action / self.n), action % self.n)
        newBoard.execute_move(move)

        return newBoard

    def getValidMoves(self, board: OthelloBoard) -> NDArray[np.int32]:
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()

        legalMoves = board.getLegalMoves()
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        
        for x, y in legalMoves:
            valids[self.n * x + y] = 1

        return np.array(valids)

    def getGameEnded(self, board: OthelloBoard) -> float:
        # TODO
        if isTerminal(board):  # e.g., max depth, or no valid moves
            reward = evaluateBoard(board)  # your framework’s score
            cost = getTotalActionCost(board)
            finalScore = reward - cost
            return finalScore
        
        return 0

        # b = Board(self.n)
        # b.pieces = np.copy(board)
        # if b.has_legal_moves():
        #     return 0
        # if b.countDiff() > 0:
        #     return 1
        # return -1

    def getScore(self, board: OthelloBoard) -> int:
        return board.countDiff()
    
    def getStringRepresentation(self, board: OthelloBoard) -> str:
        return board.getStringRepresentation()

    @staticmethod
    def display(board: OthelloBoard) -> None:
        n = board.pieces.shape[0]
        print("   ", end = "")
        for y in range(n):
            print(y, end = " ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end = "")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.squareContent[piece], end = " ")
            print("|")

        print("-----------------------")

    squareContent = {
        -1: "X",
        +0: "-",
        +1: "O"
    }
