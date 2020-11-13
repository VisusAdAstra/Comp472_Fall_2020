import sys
from queue import Queue
from queue import LifoQueue
from queue import PriorityQueue
import importlib
import xpuzzle
importlib.reload(xpuzzle)
import node
importlib.reload(node)


class Search:
    '''
    Instantiates the class, defining the start node
    '''
    def __init__(self, puzzle):
        self.start = node.Node(puzzle)


    '''

    Greedy Search Algorithm - Based in the pseudo code
    in "Artificial Intelligence: A Modern Approach - 3rd Edition"
    '''
    def greedy(self, heuristic):
        actual = self.start
        leaves = PriorityQueue()
        leaves.put((actual.costHeur(heuristic), actual))
        closed = list()
        while True:
            if leaves.empty():
                return None
            actual = leaves.get()[1]
            if actual.goalState():
                return actual
            elif actual.state.puzzle not in closed:
                closed.append(actual.state.puzzle)
                succ = actual.succ()
                while not succ.empty():
                    child = succ.get()
                    leaves.put((child.costHeur(heuristic), child))          
    
    ''' 
    A* Search Algorithm - Based in the pseudo code
    in "Artificial Intelligence: A Modern Approach - 3rd Edition"
    '''
    def aStar(self, heuristic):
        actual = self.start
        leaves = PriorityQueue()
        leaves.put((actual.costHeur(heuristic), actual))
        closed = list()
        while True:
            if leaves.empty():
                return None
            actual = leaves.get()[1]
            if actual.goalState():
                return actual
            elif actual.state.puzzle not in closed:
                closed.append(actual.state.puzzle)
                succ = actual.succ()
                while not succ.empty():
                    child = succ.get()
                    leaves.put((child.costHeur(heuristic)+child.depth, child))

                    