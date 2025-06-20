import os
import numpy as np
from NeuralNet import NeuralNet, NNetConfig
from OthelloNNet import OthelloNNet
from Game import Game
from OthelloBoard import OthelloBoard

config = NNetConfig(
    lr = 0.001,
    dropout = 0.3,
    epochs = 10,
    batch_size = 64,
    cuda = False,
    num_channels = 512,
)

class NNetWrapper(NeuralNet):
    def __init__(self, game: Game[OthelloBoard]):
        self.nnet = OthelloNNet(game, config)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples: tuple[OthelloBoard, list[float], float]):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = config.batch_size, epochs = config.epochs)

    def predict(self, board: OthelloBoard):
        """
        board: np array with board
        """
        # preparing input
        pieces = board.pieces[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(pieces, verbose = False)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))
        return pi[0], v[0]

    def saveCheckpoint(self, folder: str, filename: str):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def loadCheckpoint(self, folder: str, filename: str):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))

        self.nnet.model.load_weights(filepath)
