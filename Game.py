from typing import Generic, TypeVar
import numpy as np
from numpy.typing import NDArray

TState = TypeVar('TState')

class Game(Generic[TState]):
    """
    This class specifies the base Game class. To define your own game, subclass this class and implement the functions below.
    This works when the game is one-player and turn-based.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
        pass

    def getInitState(self) -> TState:
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form that will be the input to your neural network)
        """
        pass

    def getBoardSize(self) -> tuple[int, int]:
        """
        Returns:
            (x, y): a tuple of board dimensions
        """
        pass

    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, board: TState, action: int) -> TState:
        """
        Input:
            board: current board
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        pass

    def getValidMoves(self, board: TState) -> NDArray[np.int32]:
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.getActionSize(),
                        1 for moves that are valid from the current board,
                        0 for invalid moves
        """
        pass

    def getGameEnded(self, board: TState) -> float:
        """
        Input:
            board: current board

        Returns: 0 if game has not ended, final score (0 < x < infinity) otherwise.
        """
        pass

    def getStringRepresentation(self, board: TState) -> str:
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format. Required by MCTS for hashing.
        """
        pass
