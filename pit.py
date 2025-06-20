from Arena import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
import numpy as np
from Config import Config

"""
use this script to play any two agents against each other, or play manually with any agent.
"""

miniOthello = False  # Play in 6x6 instead of the normal 8x8.

if miniOthello:
    game = OthelloGame(6)
else:
    game = OthelloGame(8)

# all players
rp = RandomPlayer(game).play
gp = GreedyOthelloPlayer(game).play
hp = HumanOthelloPlayer(game).play

# nnet players
net = NNet(game)
if miniOthello:
    net.loadCheckpoint('./pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar')
else:
    net.loadCheckpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')

config = Config(
    numIters = 1000,
    numEps = 100,
    tempThreshold = 15,
    updateThreshold = 0.6,
    maxlenOfQueue = 200000,
    numMCTSSims = 50,
    arenaCompare = 40,
    cpuct = 1,

    checkpoint = './temp/',
    load_model = False,
    load_folder_file = ('/dev/models/8x100x50', 'best.pth.tar'),
    numItersForTrainExamplesHistory = 20
)

mcts = MCTS(game, net, config)

arena = Arena(lambda x: np.argmax(mcts.getActionProbabilities(x, temp = 0)), game, display = OthelloGame.display)

print(arena.playGames(2, verbose = True))
