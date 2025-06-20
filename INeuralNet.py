from typing import Generic, TypeVar
import numpy as np
from numpy.typing import NDArray
from IGame import IGame, TState

class INeuralNet(Generic[TState]):
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current agent, and instead only deals with
    the canonical form of the state.
    """

    def __init__(self, game: IGame[TState]):
        pass

    def train(self, examples: tuple[TState, list[float], float]) -> None:
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (state, pi, v). pi is the MCTS informed policy vector for
                      the given state, and v is its value. The examples has
                      state in its canonical form.
        """
        pass

    def predict(self, state: TState) -> tuple[NDArray[np.float64], float]:
        """
        Input:
            state: current state in its canonical form.

        Returns:
            pi: a policy vector for the current state- a numpy array of length game.getActionSize
            v: a float in [-1, 1] that gives the value of the current state
        """
        pass

    def saveCheckpoint(self, folder: str, filename: str) -> None:
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def loadCheckpoint(self, folder: str, filename: str) -> None:
        """
        Loads parameters of the neural network from folder/filename
        """
        pass

class NeuralNetConfig:
    def __init__(
        self,
        lr: float,
        dropout: float,
        epochs: int,
        batch_size: int,
        cuda: bool,
        num_channels: int,
    ):
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.num_channels = num_channels
