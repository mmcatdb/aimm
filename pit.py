import numpy as np
from Arena import Arena
from MCTS import MCTS
import alpha as impl
from Config import Config

"""
use this script to play any two agents against each other, or play manually with any agent.
"""

isMini = False  # Play in 6x6 instead of the normal 8x8.

if isMini:
    game = impl.Game(6)
else:
    game = impl.Game(8)

# all agents
rp = impl.RandomAgent(game).play
gp = impl.GreedyAgent(game).play
hp = impl.HumanAgent(game).play

# nnet agent
net = impl.NeuralNet(game)

"""
if isMini:
    net.loadCheckpoint('./pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar')
else:
    net.loadCheckpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
"""

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
    loadModel = False,
    loadFolderFile = ('/dev/models/8x100x50', 'best.pth.tar'),
    numItersForTrainExamplesHistory = 20
)

mcts = MCTS[impl.State](game, net, config)

arena = Arena(lambda x: np.argmax(mcts.getActionProbabilities(x, temp = 0)), game)

print(arena.playGames(2, verbose = True))
