#!/usr/bin/env python
# coding: utf-8

# In[245]:


import pandas as pd
import numpy as np


# In[247]:


#global data
import importlib
import xpuzzle
import node
importlib.reload(node)
importlib.reload(xpuzzle)
import search
import search2
importlib.reload(search2)
importlib.reload(search)
inputData = np.array([])
goalState=np.array([[1,2,3,4], [5,6,7,0]])


# In[248]:


#i/o
def noSolution(name, index, data):
    file = open(f"{index}_" + name + '_solution.txt', 'w')
    file.write(data[0])
    file.close()

    file = open(f"{index}_" + name + '_search.txt', 'w')
    file.write(data[1])
    file.close()


def exportData(name, index, data):
    cost = 0
    file = open(f"{index}_" + name + '_solution.txt', 'w')
    for ele in data[0].solution:
        file.write(ele)
    file.close()

    file = open(f"{index}_" + name + '_search.txt', 'w')
    for ele in data[1]:
        file.write(ele)
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
# driver for A* and GBFS test
t=xpuzzle.XPuzzle(2, 4, inputData[3]) #inputData[0] no solution
t.printPuzzle()
s=search.Search(t)

print("A*")
p  = s.aStar(1)
print(p[0])

if isinstance(p[0], node.Node):
    exportData("astar", 1, p)
else:
    noSolution("astar", 1, p)

print("GBFS")
p = s.greedyBFS(0)
print(p[0])

if isinstance(p[0], node.Node):
    exportData("gbfs", 0, p)
else:
    noSolution("gbfs", 0, p)
