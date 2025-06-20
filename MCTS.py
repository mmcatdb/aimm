from typing import Dict, Generic, TypeAlias, TypeVar
import logging
import math
import numpy as np
from numpy.typing import NDArray
from IGame import IGame, TState
from INeuralNet import INeuralNet
from Config import Config

EPSILON = 1e-8

log = logging.getLogger(__name__)

StateAction: TypeAlias = tuple[str, int]  # (state, action)

class MCTS(Generic[TState]):
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: IGame[TState], net: INeuralNet, config: Config):
        self.game = game
        self.net = net
        self.config = config
        self.Qsa: Dict[StateAction, float] = {}
        """ Q values for (s, a) (as defined in the paper). """
        self.Nsa: Dict[StateAction, int] = {}
        """ How many times the edge (s, a) was visited. """
        self.Ns: Dict[str, int] = {}
        """ How many times the state s was visited. """
        self.Ps: Dict[str, NDArray[np.float64]] = {}
        """ Initial policy vector for state s, as returned by the neural network. """

        self.Es: list[float] = {}  # stores game.getGameEnded ended for state s
        self.Vs = {}  # stores game.getValidMoves for state s

    def getActionProbabilities(self, state: TState, temp: int) -> list[float]:
        """
        This function performs numMCTSSims simulations of MCTS starting from state.

        Returns:
            probs: a policy vector where the probability of the ith action is proportional to Nsa[(s, a)] ** (1. / temp)
        """
        for i in range(self.config.numMCTSSims):
            self.search(state)

        s = self.game.getStringRepresentation(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state: TState) -> float:
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the negative of the value of the current state
        """

        s = self.game.getStringRepresentation(state)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(state)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.net.predict(state)
            valids = self.game.getValidMoves(state)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NeuralNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NeuralNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.config.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPSILON)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.getNextState(state, a)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
