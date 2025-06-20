from Arena import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import dotdict

"""
use this script to play any two agents against each other, or play manually with any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.

if mini_othello:
    game = OthelloGame(6)
else:
    game = OthelloGame(8)

# all players
rp = RandomPlayer(game).play
gp = GreedyOthelloPlayer(game).play
hp = HumanOthelloPlayer(game).play



# nnet players
n1 = NNet(game)
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar')
else:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')

args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(game, n1, args1)

arena = Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp = 0)), game, display = OthelloGame.display)

print(arena.playGames(2, verbose = True))
