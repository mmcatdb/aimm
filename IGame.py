from typing import Generic, TypeVar
import numpy as np
from numpy.typing import NDArray

TState = TypeVar('TState')

class IGame(Generic[TState]):
    """
    This class specifies the base Game class. To define your own game, subclass this class and implement the functions below.
    """
    def __init__(self):
        pass

    def getInitState(self) -> TState:
        """
        Returns: a representation of the state (ideally this is the form that will be the input to your neural network).
        """
        pass

    def getStateSize(self) -> tuple[int, int]:
        """
        Returns: a tuple of state dimensions (x, y).
        """
        pass

    def getActionSize(self) -> int:
        """
        Returns: number of all possible actions.
        """
        pass

    def getNextState(self, state: TState, action: int) -> TState:
        """
        Input:
            state: current state
            action: action taken by current agent

        Returns: state after applying action.
        """
        pass

    def getValidMoves(self, state: TState) -> NDArray[np.int32]:
        """
        Input:
            state: current state

        Returns: a binary vector of length self.getActionSize(),
                 1 for moves that are valid from the current state,
                 0 for invalid moves
        """
        pass

    def getGameEnded(self, state: TState) -> float:
        """
        Input:
            state: current state

        Returns: 0 if game has not ended, final score (0 < x < infinity) otherwise.
        """
        pass

    def getStringRepresentation(self, state: TState) -> str:
        """
        Input:
            state: current state

        Returns: a quick conversion of state to a string format. Required by MCTS for hashing.
        """
        pass
