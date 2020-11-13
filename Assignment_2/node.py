import sys
from queue import Queue
from copy import deepcopy

# ref: https://www.redblobgames.com/pathfinding/a-star/implementation.html
# https://github.com/mragankyadav/8-Puzzle-Solver
# https://github.com/rmssoares/8Puzzle-StateSpaceSearches


class Node:
    '''
	
    '''
    def __init__(self, puzzle, parent=None, move=""):
        self.state = puzzle
        self.parent = parent
        self.gn = 0
        self.hn = 0
        if parent is None:
            self.gn = 0
            self.moves = move
        else:
            temp = 1
            if move=="W":
                temp = 2
            if move=="C":
                temp = 3
            self.gn = parent.gn + temp
            self.moves = parent.moves + move

    '''
    Implement equality for queue.
    '''
    def __lt__(self, other):
        return self.hn < other.hn

    '''
    Checks if the Node's state is a goal state.
    '''
    def goalState(self):
        return self.state.checkPuzzle()

    '''
    Generates the node's children states.
    '''
    def succ(self):
        succs = Queue()
        for m in self.state.moves:
            p = deepcopy(self.state)
            p.doMove(m)
            if p.zero is not self.state.zero:
                succs.put(Node(p, self, m))
        return succs

    '''
    Select heuristics
    '''
    def costHeur(self, heuristic):
        if heuristic is 0:
            return self.hammingDistance()
        else:
            self.hn = self.manhattanDistance()
            return self.hn

    '''
    First heuristic - hammingDistance
    Every time there's a tile in the wrong place, we
    add 1 to the result. Do not count zero to maintain admissible
    '''
    def hammingDistance(self):
        result = [0, 0]
        count = 1
        for i in range(0, self.state.row):
            for j in range(0, self.state.col):
                if self.state.zero != (i, j) and self.state.puzzle[i][j] != (count % (self.state.row*self.state.col)):
                    result[0] += 1
                count += 1
        count = 1
        for j in range(0, self.state.col):
            for i in range(0, self.state.row):
                if self.state.zero != (i, j) and self.state.puzzle[i][j] != (count % (self.state.row*self.state.col)):
                    result[1] += 1
                count += 1
        if result[0] < result[1]:
            return result[0]
        else:
            return result[1]

    '''
    Second heuristic - manhattanDistance
    Distance of wrong tiles to their correct position. 
    r = n-1 / col
    c = n-1 % col
    r = n-1 % row
    c = n-1 / row
    '''
    def manhattanDistance(self):
        result = [0, 0]
        distance = [0, 0]
        for i in range(0, self.state.row):
            for j in range(0, self.state.col):
                index = self.state.puzzle[i][j] - 1
                if index == -1:
                    distance[0] = (self.state.row-1-i)+(self.state.col-1-j)
                    distance[1] = (self.state.row-1-i)+(self.state.col-1-j)
                else:
                    distance[0] = abs(i-(index//self.state.col)) + \
                        abs(j-(index % self.state.col))
                    distance[1] = abs(i-(index % self.state.row)) + \
                        abs(j-(index//self.state.row))
                result[0] += distance[0]
                result[1] += distance[1]
        if result[0] < result[1]:
            return result[0]
        else:
            return result[1]

    '''
    obtain the moves from the
    starting state to this specific one.
    '''
    def record(self):
        return str(self.moves)

    '''
    When printing the node, we obtain the moves from the
    starting state to this specific one.
    '''
    def __str__(self):
        return str(self.moves)


