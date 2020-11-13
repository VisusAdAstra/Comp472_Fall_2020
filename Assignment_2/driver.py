#!/usr/bin/env python
# coding: utf-8

# In[245]:


import pandas as pd
import numpy as np


# In[247]:


#global data
import importlib
import xpuzzle
importlib.reload(xpuzzle)
import search
import search2
importlib.reload(search2)
importlib.reload(search)
inputData = np.array([])
goalState=np.array([[1,2,3,4], [5,6,7,0]])


# In[248]:


#i/o
def exportData(frontier, time):

    moves = 1
    file = open('output.txt', 'w')
    file.write("path_to_goal: " + str(moves))
    file.write("\ncost_of_path: " + str(len(moves)))
    file.write("\nnodes_expanded: " + str(nodes_expanded))
    file.write("\nfringe_size: " + str(len(frontier)))
    file.write("\nmax_fringe_size: " + str(max_frontier_size))
    file.write("\nsearch_depth: " + str(goal_node.depth))
    file.write("\nmax_search_depth: " + str(max_search_depth))
    file.write("\nrunning_time: " + format(time, '.8f'))
    file.close()


def importData(file_name):

    data = []
    file = open(file_name, 'r')
    for line in file:
        data.append([int(x) for x in line.split(" ")])

    data = np.array(data)
    board_len = len(data[0])
    board_side = int(board_len ** 0.5)
    return data
#C:\\Users\\Chun\\Documents\\Comp472_Fall_2020\\Assignment_2\\samplePuzzles.txt
inputData = importData('samplePuzzles.txt')
print(inputData)

# In[253]:

f=search2.Search2()
f.uniform_cost_search([2, 6, 3, 4, 5, 7, 0, 1], 2, 4, [1, 2, 3, 4, 5, 6, 7, 0])

t=xpuzzle.XPuzzle(2, 4, inputData[3]) #inputData[0] no solution
t.doMove("C")
t.printPuzzle()

s=search.Search(t)
p = s.bestFirst()
p.state.printPuzzle()
p = s.aStar(1)
p.state.printPuzzle()
p.state.manhattanDistance()


# In[ ]:




