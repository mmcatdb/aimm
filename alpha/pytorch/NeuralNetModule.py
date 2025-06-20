import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from IGame import IGame
from INeuralNet import NeuralNetConfig
from State import State

class NeuralNetModule(nn.Module):
    def __init__(self, game: IGame[State], config: NeuralNetConfig):
        # game params
        self.stateX, self.stateY = game.getStateSize()
        self.action_size = game.getActionSize()
        self.config = config

        super(NeuralNetModule, self).__init__()
        self.conv1 = nn.Conv2d(1, config.num_channels, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride = 1)
        self.conv4 = nn.Conv2d(config.num_channels, config.num_channels, 3, stride = 1)

        self.bn1 = nn.BatchNorm2d(config.num_channels)
        self.bn2 = nn.BatchNorm2d(config.num_channels)
        self.bn3 = nn.BatchNorm2d(config.num_channels)
        self.bn4 = nn.BatchNorm2d(config.num_channels)

        self.fc1 = nn.Linear(config.num_channels * (self.stateX - 4) * (self.stateY - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s) -> tuple[Tensor, Tensor]:
        #                                                           s: batch_size x stateX x stateY
        s = s.view(-1, 1, self.stateX, self.stateY)                # batch_size x 1 x stateX x stateY
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x stateX x stateY
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x stateX x stateY
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (stateX - 2) x (stateY - 2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (stateX - 4) x (stateY - 4)
        s = s.view(-1, self.config.num_channels * (self.stateX - 4) * (self.stateY - 4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p = self.config.dropout, training = self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p = self.config.dropout, training = self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim = 1), torch.tanh(v)
