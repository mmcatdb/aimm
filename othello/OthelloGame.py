from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .OthelloBoard import OthelloBoard
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

State: TypeAlias = NDArray[np.int32]

class OthelloGame(Game):
    def __init__(self, n):
        self.n = n

    def getInitState(self) -> State:
        # return initial board (numpy board)
        b = OthelloBoard(self.n)
        return np.array(b.pieces)

    def getBoardSize(self) -> tuple[int, int]:
        # (a, b) tuple
        return (self.n, self.n)

    def getActionSize(self) -> int:
        # return number of actions
        return self.n * self.n + 1

    def getNextState(self, board, action) -> list[list[int]]:
        # if player takes action on board, return next board
        # action must be a valid move
        if action == self.n * self.n:
            return board
        
        b = OthelloBoard(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move)

        return b.pieces

    def getValidMoves(self, board) -> NDArray[np.int32]:
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = OthelloBoard(self.n)
        b.pieces = np.copy(board)

        legalMoves =  b.get_legal_moves()
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        
        for x, y in legalMoves:
            valids[self.n * x + y] = 1

        return np.array(valids)

    def getGameEnded(self, board) -> float:
        # TODO
        if is_terminal(board):  # e.g., max depth, or no valid moves
            reward = evaluate_board(board)  # your framework’s score
            cost = get_total_action_cost(board)
            final_score = reward - cost
            return final_score
        
        return 0

        # b = Board(self.n)
        # b.pieces = np.copy(board)
        # if b.has_legal_moves():
        #     return 0
        # if b.countDiff() > 0:
        #     return 1
        # return -1

    def getScore(self, board) -> int:
        b = OthelloBoard(self.n)
        b.pieces = np.copy(board)
        return b.countDiff()
    
    def stringRepresentation(self, board) -> str:
        return board.tostring()

    def stringRepresentationReadable(self, board) -> str:
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def display(board) -> None:
        n = board.shape[0]
        print("   ", end = "")
        for y in range(n):
            print(y, end = " ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end = "")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end = " ")
            print("|")

        print("-----------------------")
