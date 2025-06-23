class Config:
    def __init__(
        self,
        numIters: int,
        numEps: int,
        tempThreshold: int,
        updateThreshold: float,
        maxlenOfQueue: int,
        numMCTSSims: int,
        arenaCompare: int,
        cpuct: float,

        checkpoint: str,
        loadModel: bool,
        loadFolderFile: tuple,
        numItersForTrainExamplesHistory: int
    ):
        self.numIters = numIters
        """ Number of complete self-play games to simulate during a new iteration. """

        self.numEps = numEps

        self.tempThreshold = tempThreshold

        self.updateThreshold = updateThreshold
        """ During arena playoff, new neural net will be accepted if threshold or more of games are won. """

        self.maxlenOfQueue = maxlenOfQueue
        """ Number of game examples to train the neural networks. """

        self.numMCTSSims = numMCTSSims
        """ Number of games moves for MCTS to simulate. """

        self.arenaCompare = arenaCompare
        """ Number of games to play during arena play to determine if new net will be accepted. """

        self.cpuct = cpuct

        self.checkpoint = checkpoint
        
        self.loadModel = loadModel
        
        self.loadFolderFile = loadFolderFile
        
        self.numItersForTrainExamplesHistory = numItersForTrainExamplesHistory
