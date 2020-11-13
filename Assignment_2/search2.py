#!/usr/bin/env python
# coding: utf-8
import sys
from queue import Queue
from queue import LifoQueue
from queue import PriorityQueue
import importlib
import xpuzzle
import numpy
importlib.reload(xpuzzle)
import node
importlib.reload(node)


class Search2:

    columns = 0
    rows = 0
    move_adjacent = 1
    move_diagonal = 3
    move_wrapping = 5

    def get_right_node(self, array, index):
        return array[index + 1], Search2.move_adjacent

    def get_left_node(self, array, index):
        return array[index - 1], Search2.move_adjacent

    def get_node_below(self, array, index):
        return array[(index + Search2.columns)], Search2.move_adjacent
    
    def get_node_above(self, array, index):
        return array[(index - Search2.columns)], Search2.move_adjacent

    #get all the neighboring nodes of a node
    def get_current_node_neighbors(self, current_node, array):
        neighbors = []
        nodeIndex = array.index(current_node)

        #if 0 in top left corner
        if nodeIndex == 0:
            neighbors.append(self.get_right_node(array, nodeIndex))
            neighbors.append(self.get_node_below(array, nodeIndex))
            #diagonal below
            neighbors.append((array[nodeIndex + Search2.columns + 1], Search2.move_diagonal))
            #wrapping
            neighbors.append((array[Search2.columns - 1], Search2.move_wrapping))
            #diagonal other side
            neighbors.append(((array[len(array) - 1] ), Search2.move_diagonal))

        #if 0 in top right corner
        elif nodeIndex == (Search2.columns - 1):
            neighbors.append(self.get_left_node(array, nodeIndex))
            neighbors.append(self.get_node_below(array, nodeIndex))
            #diagonal below
            neighbors.append((array[nodeIndex + Search2.columns - 1], Search2.move_diagonal))
            #wrapping
            neighbors.append((array[0], Search2.move_wrapping))
            #diagonal other side
            neighbors.append((array[Search2.columns * (Search2.rows - 1)], Search2.move_diagonal))

        #if 0 in bottom left corner   
        elif nodeIndex == (Search2.columns * (Search2.rows - 1)):
            neighbors.append(self.get_node_above(array, nodeIndex))
            neighbors.append(self.get_right_node(array, nodeIndex))
            #diagonal above
            neighbors.append((array[nodeIndex - Search2.columns + 1], Search2.move_diagonal))
            #wrapping
            neighbors.append((array[len(array) - 1], Search2.move_wrapping))
            #diagonal other side
            neighbors.append((array[Search2.columns - 1], Search2.move_diagonal))

        #if 0 in bottom right corner   
        elif nodeIndex == (len(array) - 1):
            neighbors.append(self.get_node_above(array, nodeIndex))
            neighbors.append(self.get_left_node(array, nodeIndex))
            #diagonal above
            neighbors.append((array[Search2.columns * (Search2.rows- 1) - 2], Search2.move_diagonal))
            #wrapping
            neighbors.append((array[(Search2.rows - 1) * Search2.columns], Search2.move_wrapping))
            #diagonal other side
            neighbors.append((array[0], Search2.move_diagonal))

        else:
            #node is at the top row
            if nodeIndex < Search2.columns :
                neighbors.append(self.get_node_below(array, nodeIndex))
                neighbors.append(self.get_right_node(array, nodeIndex))
                neighbors.append(self.get_left_node(array, nodeIndex))
            #node is at the bottom row
            elif nodeIndex > (((Search2.rows - 1) * Search2.columns) - 1):
                neighbors.append(self.get_node_above(array, nodeIndex))
                neighbors.append(self.get_right_node(array, nodeIndex))
                neighbors.append(self.get_left_node(array, nodeIndex))
            #node is on left column in the middle
            elif (nodeIndex % Search2.columns) == 0 :
                neighbors.append(self.get_node_above(array, nodeIndex))
                neighbors.append(self.get_node_below(array, nodeIndex))
                neighbors.append(self.get_right_node(array, nodeIndex))
            #node is on right column in the middle
            elif ((nodeIndex + 1) % Search2.columns) == 0 :
                neighbors.append(self.get_node_above(array, nodeIndex))
                neighbors.append(self.get_node_below(array, nodeIndex))
                neighbors.append(self.get_left_node(array, nodeIndex))
            #node is in middle
            else:
                neighbors.append(self.get_node_below(array, nodeIndex))
                neighbors.append(self.get_right_node(array, nodeIndex))
                neighbors.append(self.get_node_above(array, nodeIndex))
                neighbors.append(self.get_left_node(array, nodeIndex))
            

        return neighbors

    def uniform_cost_search(self, array, arrayRows, arrayColumns, goal):    
        #sets the number of columns and rows (by default, it should be 2 rows and 4 columns)
        Search2.columns = arrayColumns
        Search2.rows = arrayRows
        
        #contains the explored nodes
        explored_nodes = []  
        
        #if start == goal:    
         #   return  explored_nodes    
        
        #starting node (original array structure, swapping value, total cost, previous value)
        t = (array, 0, 0, None)
        
        #contains all the nodes
        nodes = [] 
        nodes.append(t)

        #weight total
        old_cost = 0

        while len(nodes) > 0:    
            #sort nodes based on cost
            nodes.sort(key=lambda x: x[2])
            current_node = nodes.pop(0)
            neighbours = self.get_current_node_neighbors(current_node[1], array)
            #Structure of neighbor tuple: (value, cost)
            #Structure of node: (current state of the array, value, total cost, previous value)
            for neighbour in neighbours:  
                #Ensures that the node that you're swapping with is not the one that you previously swapped with
                if(neighbour[0] != current_node[3]):

                    indexOfZero = current_node[0].index(0)
                    indexOfValue = current_node[0].index(neighbour[0])
                    
                    copiedArray = current_node[0].copy()

                    #swap the values
                    copiedArray[indexOfValue] = 0
                    copiedArray[indexOfZero] = neighbour[0]

                    #t = (array, neighbour[0], neighbour[1] + current_node[1])
                    m = (current_node[1], current_node[2])
                    explored_nodes.append(m)

                    #if the swapped array is equal to the goal array
                    if numpy.array_equal(copiedArray, goal):
                        if ((neighbour[1] + current_node[2]) > old_cost or old_cost == 0):
                            old_cost = neighbour[1] + current_node[2]

                            print(copiedArray)
                            print(old_cost)

                            return old_cost
                    #else, if it's not an explored node, add it to the list of nodes that will be traversed
                    elif (neighbour not in explored_nodes):
                        t = (copiedArray, neighbour[0], (neighbour[1] + current_node[2]), neighbour[0])
                        nodes.append(t) 
                        
        return None, None  
    
