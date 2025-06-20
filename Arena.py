import logging
from typing import Callable, Generic
from tqdm import tqdm
from IGame import IGame, TState

log = logging.getLogger(__name__)

class Arena(Generic[TState]):
    """
    An Arena class where an agent can be tested.
    """

    def __init__(self, agent: Callable, game: IGame[TState]):
        """
        Input:
            agent: function that takes state as input and returns action
            game: Game object
        """
        self.agent = agent
        self.game = game

    @staticmethod
    def testAgent(agentFunction: Callable[[TState], int], game: IGame[TState], iterations: int) -> float:
        """
        Test the agent on the given amount of iterations.
        Returns the normalized score (total score divided by num).
        """
        arena = Arena[TState](agentFunction, game)

        score = 0
        for _ in tqdm(range(iterations), desc = "Arena.testAgent (1)"):
            score += arena.__testOnce()

        return score / iterations

    def __testOnce(self) -> float:
        """
        Executes one episode of a game. Returns the final score (0 < x < infinity).
        """
        state = self.game.getInitState()
        it = 0

        while self.game.getGameEnded(state) == 0:
            it += 1

            action = self.agent(state)
            valids = self.game.getValidMoves(state)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            state = self.game.getNextState(state, action)
        
        return self.game.getGameEnded(state)
