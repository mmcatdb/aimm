import logging

from typing import Callable
from tqdm import tqdm
from Game import Game

log = logging.getLogger(__name__)

class Arena():
    """
    An Arena class where an agent can be tested.
    """

    def __init__(self, player: Callable, game: Game):
        """
        Input:
            player: function that takes board as input and returns action
            game: Game object
            display: a function that takes board as input and prints it (e.g. display in othello/OthelloGame).
                Is necessary for verbose mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting human players/other baselines with each other.
        """
        self.player = player
        self.game = game

    @staticmethod
    def testAgent(agentFunction: Callable[[any], int], game: Game, iterations: int) -> float:
        """
        Test the agent on the given amount of iterations.
        Returns the normalized score (total score divided by num).
        """

        arena = Arena(agentFunction, game)

        score = 0
        for _ in tqdm(range(iterations), desc = "Arena.testAgent (1)"):
            score += arena.testOnce()

        return score / iterations

    def testOnce(self) -> float:
        """
        Executes one episode of a game. Returns the final score (0 < x < infinity).
        """
        board = self.game.getInitState()
        it = 0

        while self.game.getGameEnded(board) == 0:
            it += 1

            action = self.player(board)
            valids = self.game.getValidMoves(board)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            board = self.game.getNextState(board, action)
        
        return self.game.getGameEnded(board)
