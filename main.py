import logging

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper
from Config import Config

log = logging.getLogger(__name__)

coloredlogs.install(level = 'INFO')  # Change this to DEBUG to see more info.

args = Config(
    numIters = 1000,
    numEps = 100,
    tempThreshold = 15,
    updateThreshold = 0.6,
    maxlenOfQueue = 200000,
    numMCTSSims = 25,
    arenaCompare = 40,
    cpuct = 1,

    checkpoint = './temp/',
    load_model = False,
    load_folder_file = ('/dev/models/8x100x50', 'best.pth.tar'),
    numItersForTrainExamplesHistory = 20
)

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
