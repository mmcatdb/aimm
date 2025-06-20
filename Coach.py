import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS
from Game import Game
from NeuralNet import NeuralNet
from Config import Config
from typing import Any

log = logging.getLogger(__name__)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined in Game and NeuralNet.
    """

    def __init__(self, game: Game, net: NeuralNet, config: Config):
        self.game = game
        self.net = net
        self.prevNet = self.net.__class__(self.game)  # the competitor network
        self.config = config
        self.mcts = MCTS(self.game, self.net, self.config)
        self.trainExamplesHistory = []  # history of examples from config.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def learn(self) -> None:
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.config.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen = self.config.maxlenOfQueue)

                for _ in tqdm(range(self.config.numEps), desc = "Self Play"):
                    self.mcts = MCTS(self.game, self.net, self.config)  # reset search tree
                    iterationTrainExamples.append(self.__executeEpisode())

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.config.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i - 1)  
            self.__saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.net.saveCheckpoint(folder = self.config.checkpoint, filename = 'temp.pth.tar')
            self.prevNet.loadCheckpoint(folder = self.config.checkpoint, filename = 'temp.pth.tar')
            prevMcts = MCTS(self.game, self.prevNet, self.config)

            self.net.train(trainExamples)
            mcts = MCTS(self.game, self.net, self.config)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            prevScore = Arena.testAgent(lambda x: np.argmax(prevMcts.getActionProbabilities(x, temp = 0), self.game), self.config.arenaCompare)
            score = Arena.testAgent(lambda x: np.argmax(mcts.getActionProbabilities(x, temp = 0), self.game), self.config.arenaCompare)
                             
            log.info('NEW/PREV SCORE : %f / %f' % (score, prevScore))
            if score / prevScore > self.config.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.net.loadCheckpoint(folder = self.config.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.net.saveCheckpoint(folder = self.config.checkpoint, filename=self.__getCheckpointFile(i))
                self.net.saveCheckpoint(folder = self.config.checkpoint, filename='best.pth.tar')

    def __executeEpisode(self) -> tuple[Any, list[float], float]:
        """
        This function executes one episode of self-play.
        As the game is played, each turn is added as a training example to trainExamples.
        The game is played till the game ends. After the game ends, the outcome of the game is used to assign values to each example in trainExamples.

        It uses a temp = 1 if episodeStep < tempThreshold, and thereafter
        uses temp = 0.

        Returns:
            trainExamples: a list of examples of the form (board, pi, v)
                           pi is the MCTS informed policy vector, v is +1 if the player eventually won the game, else -1.
        """
        board = self.game.getInitState()
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.config.tempThreshold)

            pi = self.mcts.getActionProbabilities(board, temp = temp)

            action = np.random.choice(len(pi), p = pi)
            board = self.game.getNextState(board, action)

            r = self.game.getGameEnded(board)

            if r != 0:
                return (board, pi, r)

    def __saveTrainExamples(self, iteration: int) -> None:
        folder = self.config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.__getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed
    
    def __getCheckpointFile(self, iteration: int) -> str:
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def loadTrainExamples(self) -> None:
        modelFile = os.path.join(self.config.load_folder_file[0], self.config.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
