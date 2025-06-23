import logging
import coloredlogs
from Coach import Coach
from Config import Config
import alpha as impl

log = logging.getLogger(__name__)

coloredlogs.install(level = 'INFO')  # Change this to DEBUG to see more info.

config = Config(
    numIters = 1000,
    numEps = 100,
    tempThreshold = 15,
    updateThreshold = 0.6,
    maxlenOfQueue = 200000,
    numMCTSSims = 25,
    arenaCompare = 40,
    cpuct = 1,

    checkpoint = './temp/',
    loadModel = False,
    loadFolderFile = ('/dev/models/8x100x50', 'best.pth.tar'),
    numItersForTrainExamplesHistory = 20
)

def main():
    log.info('Loading %s...', impl.Game.__name__)
    g = impl.Game(6)

    log.info('Loading %s...', impl.NeuralNet.__name__)
    net = impl.NeuralNet(g)

    if config.loadModel:
        log.info('Loading checkpoint "%s/%s"...', config.loadFolderFile[0], config.loadFolderFile[1])
        net.loadCheckpoint(config.loadFolderFile[0], config.loadFolderFile[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach[impl.State](g, net, config)

    if config.loadModel:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
