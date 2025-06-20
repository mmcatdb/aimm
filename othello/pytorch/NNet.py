import os
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from NeuralNet import NeuralNet, NNetConfig
import torch
from torch import Tensor, FloatTensor
import torch.optim as optim
from OthelloNNet import OthelloNNet
from OthelloBoard import OthelloBoard
from Game import Game

config = NNetConfig(
    lr = 0.001,
    dropout = 0.3,
    epochs = 10,
    batch_size = 64,
    cuda = torch.cuda.is_available(),
    num_channels = 512,
)

class NNetWrapper(NeuralNet[OthelloBoard]):
    def __init__(self, game: Game[OthelloBoard]):
        self.nnet = OthelloNNet(game, config)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if config.cuda:
            self.nnet.cuda()

    def train(self, examples: tuple[OthelloBoard, list[float], float]) -> None:
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(config.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / config.batch_size)

            t = tqdm(range(batch_count), desc = 'Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size = config.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = FloatTensor(np.array(boards).astype(np.float64))
                target_pis = FloatTensor(np.array(pis))
                target_vs = FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if config.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.__lossPi(target_pis, out_pi)
                l_v = self.__lossV(target_vs, out_v)
                # TODO Adapt this for regression instead of classification
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi = pi_losses, Loss_v = v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board: OthelloBoard) -> tuple[NDArray[np.float64], float]:
        """
        board: np array with board
        """
        # preparing input
        pieces = board.pieces
        pieces = FloatTensor(pieces.astype(np.float64))
        if config.cuda:
            pieces = pieces.contiguous().cuda()
        pieces = pieces.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(pieces)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def __lossPi(self, targets: Tensor | FloatTensor, outputs: Tensor | FloatTensor) -> Tensor:
        return -torch.sum(targets * outputs) / targets.size()[0]

    def __lossV(self, targets: Tensor | FloatTensor, outputs: Tensor | FloatTensor) -> Tensor:
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def saveCheckpoint(self, folder: str, filename: str) -> None:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def loadCheckpoint(self, folder: str, filename: str) -> None:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if config.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location = map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
