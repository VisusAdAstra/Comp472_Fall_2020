import sys
from queue import Queue
from queue import LifoQueue
from queue import PriorityQueue
import time
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
        self.limit = 60


    '''
    Greedy Best First Search Algorithm - Based in the pseudo code
    in "Artificial Intelligence: A Modern Approach - 3rd Edition"
    '''
    def greedyBFS(self, heuristic):
        actual = self.start
        leaves = PriorityQueue()
        leaves.put((actual.costHeur(heuristic), actual))
        closed = list()
        start_time = time.time()
        while True:
            if (time.time() - start_time > self.limit):
                return ("no solution", "no solution")
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
        start_time = time.time()
        while True:
            if (time.time() - start_time > self.limit):
                return ("no solution", "no solution")
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
                    leaves.put((child.costHeur(heuristic)+child.gn, child))

