import sys
from queue import Queue
from queue import LifoQueue
from queue import PriorityQueue
import importlib
import xpuzzle
import node
importlib.reload(xpuzzle)
importlib.reload(node)


class Search:

    '''
    Instantiates the class, defining the start node
    '''
    def __init__(self, puzzle):
        self.start = node.Node(puzzle)

    '''
    Best First Search Algorithm - Based in the pseudo code
    in "Artificial Intelligence: A Modern Approach - 3rd Edition"
    '''
    def bestFirst(self):
        closed = list()
        leaves = Queue()
        leaves.put(self.start)
        while True:
            if leaves.empty():
                return None
            actual = leaves.get()
            if actual.goalState():
                return actual
            elif actual.state.puzzle not in closed:
                closed.append(actual.state.puzzle)
                succ = actual.succ()
                while not succ.empty():
                    leaves.put(succ.get())
    
    '''
    Iterative Deepening Search Algorithm - Based in the pseudo code
    in "Artificial Intelligence: A Modern Approach - 3rd Edition"
    '''
    def iterativeDeepening(self):
        depth = 0
        result = None
        while result == None:
            result = self.depthLimited(depth)
            depth +=1
        return result
    
    '''
    Depth Limited Search Algorithm - Based in the pseudo code
    in "Artificial Intelligence: A Modern Approach - 3rd Edition"
    '''
    def depthLimited(self, depth):
        leaves = LifoQueue()
        leaves.put(self.start)
        while True:
            if leaves.empty():
                return None
            actual = leaves.get()
            if actual.goalState():
                return actual
            elif actual.depth is not depth:
                succ = actual.succ()
                while not succ.empty():
                    leaves.put(succ.get())

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
    def aSearch(self, heuristic):
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

                    