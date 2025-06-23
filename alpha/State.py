from typing import Any, Generator, TypeAlias
import numpy as np

Position: TypeAlias = tuple[int, int]  # A position on the state is represented as a tuple of (x, y) coordinates.

class State():
    """
    first dim is column , 2nd is row:
        pieces[1][7] is the square in column 2,
        at the opposite end of the state in row 8.
    Squares are stored and manipulated as (x, y) tuples.
    x is the column, y is the row.
    """

    # list of all 8 directions on the state, as (x, y) offsets
    __directions = [ (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1) ]

    def __init__(self, n):
        "Set up initial state configuration."

        self.n = n
        # Create the empty state array.
        pieces: list[list[int]] = [None] * self.n
        for i in range(self.n):
            pieces[i] = [0] * self.n

        # Set up the initial 4 pieces.
        pieces[int(self.n / 2) - 1][int(self.n / 2)] = 1
        pieces[int(self.n / 2)][int(self.n / 2)-1] = 1
        pieces[int(self.n / 2) - 1][int(self.n / 2)-1] = -1
        pieces[int(self.n / 2)][int(self.n / 2)] = -1

        self.pieces = np.array(pieces)

    @staticmethod
    def copy(state: 'State') -> 'State':
        """ Returns a copy of the given state. """
        newState = State(state.n)
        newState.pieces = np.copy(state.pieces)
        return newState

    # add [][] indexer syntax to the state
    def __getitem__(self, index: int) -> list[int]:
        return self.pieces[index]

    def countDiff(self) -> int:
        """ Counts the # pieces of the given color """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 1:
                    count += 1

        return count

    def getLegalMoves(self) -> list[Position]:
        """ Returns all the legal moves for the given color. """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 1:
                    newMoves = self.__getMovesForSquare((x, y))
                    moves.update(newMoves)

        return list(moves)

    def hasLegalMoves(self) -> bool:
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 1:
                    newMoves = self.__getMovesForSquare((x, y))
                    if len(newMoves) > 0:
                        return True

        return False

    def __getMovesForSquare(self, square: Position) -> list[Position]:
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3, 4) and it contains a black piece,
        and (3, 5) and (3, 6) contain white pieces, and (3, 7) is empty, one
        of the returned moves is (3, 7) because everything from there to (3, 4)
        is flipped.
        """
        (x, y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty source squares.
        if color == 0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self.__discoverMove(square, direction)
            if move:
                # print(square, move, direction)
                moves.append(move)

        # return the generated move list
        return moves

    def executeMove(self, move: Position) -> None:
        """Perform the given move on the state; flips pieces as necessary.
        color gives the color pf the piece to play (1 = white, -1 = black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        flips = [flip for direction in self.__directions
                      for flip in self.__getFlips(move, direction)]
        assert len(list(flips)) > 0
        for x, y in flips:
            #print(self[x][y], color)
            self[x][y] = 1

    def __discoverMove(self, origin: Position, direction: Position) -> Position:
        """ Returns the endpoint for a legal move, starting at the given origin, moving by the given increment. """
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in State.__incrementMove(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x, y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip", x, y)
                flips.append((x, y))

    def __getFlips(self, origin: Position, direction: Position) -> list[Position]:
        """ Gets the list of flips for a vertex and direction to use with the executeMove function. """
        #initialize variables
        flips = [origin]

        for x, y in State.__incrementMove(origin, direction, self.n):
            #print(x, y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -1:
                flips.append((x, y))
            elif self[x][y] == 1 and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    def __incrementMove(move: Position, direction: Position, n: int) -> Generator[Position, Any, None]:
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0] + direction[0], move[1] + direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0 <= move[0] and move[0] < n and 0 <= move[1] and move[1] < n:
            yield move
            move = list(map(sum, zip(move, direction)))
            #move = (move[0] + direction[0], move[1] + direction[1])

    def getStringRepresentation(self) -> str:
        """ Returns a string representation of the state. """
        return self.pieces.tostring()
    