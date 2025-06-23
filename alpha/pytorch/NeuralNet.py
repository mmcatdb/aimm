import os
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from INeuralNet import INeuralNet, NeuralNetConfig
import torch
from torch import Tensor, FloatTensor
import torch.optim as optim
from NeuralNetModule import NeuralNetModule
from State import State
from IGame import IGame

config = NeuralNetConfig(
    lr = 0.001,
    dropout = 0.3,
    epochs = 10,
    batchSize = 64,
    cuda = torch.cuda.is_available(),
    numChannels = 512,
)

class NeuralNet(INeuralNet[State]):
    def __init__(self, game: IGame[State]):
        self.module = NeuralNetModule(game, config)
        self.stateSize = game.getStateSize()
        self.action_size = game.getActionSize()

        if config.cuda:
            self.module.cuda()

    def train(self, examples: tuple[State, list[float], float]) -> None:
        """
        examples: list of examples, each example is of form (state, pi, v)
        """
        optimizer = optim.Adam(self.module.parameters())

        for epoch in range(config.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.module.train()
            lossesPi = AverageMeter()
            lossesV = AverageMeter()

            batchCount = int(len(examples) / config.batchSize)

            t = tqdm(range(batchCount), desc = 'Training Net')
            for _ in t:
                sampleIds = np.random.randint(len(examples), size = config.batchSize)
                states, pis, vs = list(zip(*[examples[i] for i in sampleIds]))
                states = FloatTensor(np.array(states).astype(np.float64))
                targetPis = FloatTensor(np.array(pis))
                targetVs = FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if config.cuda:
                    states, targetPis, targetVs = states.contiguous().cuda(), targetPis.contiguous().cuda(), targetVs.contiguous().cuda()

                # compute output
                outPi, outV = self.module(states)
                lossPi = self.__lossPi(targetPis, outPi)
                lossV = self.__lossV(targetVs, outV)
                # TODO Adapt this for regression instead of classification
                totalLoss = lossPi + lossV

                # record loss
                lossesPi.update(lossPi.item(), states.size(0))
                lossesV.update(lossV.item(), states.size(0))
                t.set_postfix(Loss_pi = lossesPi, Loss_v = lossesV)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                totalLoss.backward()
                optimizer.step()

    def predict(self, state: State) -> tuple[NDArray[np.float64], float]:
        """
        state: np array with state
        """
        # preparing input
        pieces = state.pieces
        pieces = FloatTensor(pieces.astype(np.float64))
        if config.cuda:
            pieces = pieces.contiguous().cuda()
        pieces = pieces.view(1, self.stateSize[0], self.stateSize[1])
        self.module.eval()
        with torch.no_grad():
            pi, v = self.module(pieces)

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
            'state_dict': self.module.state_dict(),
        }, filepath)

    def loadCheckpoint(self, folder: str, filename: str) -> None:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        mapLocation = None if config.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location = mapLocation)
        self.module.load_state_dict(checkpoint['state_dict'])

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
