import numpy as np
from numpy.typing import NDArray
from Game import Game

class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game: Game):
        pass

    def train(self, examples) -> None:
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        pass

    def predict(self, board) -> tuple[NDArray[np.float64], float]:
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length game.getActionSize
            v: a float in [-1, 1] that gives the value of the current board
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

class NNetConfig:
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
