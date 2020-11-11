import sys
from queue import Queue
from copy import deepcopy


class Node:
    '''
    Instantiates the node. Only the passing argument
    "puzzle" is necessary for the creating of a Node.
    So, we've given the other two values a default value.
    '''
    def __init__(self, puzzle, parent=None, move=""):
        self.state = puzzle
        self.parent = parent
        self.depth = 0
        self.gn = 0
        self.hn = 0
        if parent is None:
            self.depth = 0
            self.moves = move
        else:
            self.depth = parent.depth+1
            self.moves = parent.moves + move
    
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
            return self.nWrongTiles()
        else:
            self.hn = self.manhattanDistance()
            return self.hn

    '''
    First heuristic - number of wrong tiles
    Every time there's a tile in the wrong place, we
    add 1 to the result. Heavily inspired in the
    puzzle.checkPuzzle() loop.
    '''
    def nWrongTiles(self):
        result = 0
        count = 1
        for i in range(0,self.state.row):
            for j in range(0,self.state.col):
                if self.state.puzzle[i][j]!=(count%(self.state.row*self.state.col)):
                    result += 1
                count+=1
        return result

    '''
    Second heuristic - distance of wrong tiles to their
    right position. After a little bit of scheming, came
    the mathematical conclusion that:
    x = n-1 %3
    y = n-1 /3
    which concluded into the following result.
    '''
    def manhattanDistance(self):
        result = 0
        count = 1
        for i in range(0,self.state.row):
            for j in range(0,self.state.col):
                index = self.state.puzzle[i][j] - 1
                distance = (2-i)+(2-j) if index == -1 else abs(i-(index/self.state.row))+abs(j-(index%self.state.col))
                result += distance
                count+=1
        return result
    
    '''
    When printing the node, we obtain the moves from the
    starting state to this specific one.
    '''
    def __str__(self):
        return str(self.moves)


