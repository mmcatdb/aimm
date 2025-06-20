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
    load_model = False,
    load_folder_file = ('/dev/models/8x100x50', 'best.pth.tar'),
    numItersForTrainExamplesHistory = 20
)

def main():
    log.info('Loading %s...', impl.Game.__name__)
    g = impl.Game(6)

    log.info('Loading %s...', impl.NeuralNet.__name__)
    net = impl.NeuralNet(g)

    if config.load_model:
        log.info('Loading checkpoint "%s/%s"...', config.load_folder_file[0], config.load_folder_file[1])
        net.loadCheckpoint(config.load_folder_file[0], config.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach[impl.State](g, net, config)

    if config.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
