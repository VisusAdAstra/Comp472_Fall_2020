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
def exportData(name, index, data, heu=-1):
    subname = "-h" + str(heu) if heu > -1 else ""
    solution = f"output/{index}_{name}{subname}" + '_solution.txt'
    search = f"output/{index}_{name}{subname}" + '_search.txt'
    file1 = open(solution, 'w')
    file2 = open(search, 'w')
    if isinstance(data[0], node.Node):
        for ele in data[0].solution:
            file1.write(ele)
        file1.write(f"{data[0].gn} {data[2]}")
        for ele in data[1]:
            file2.write(ele)
    else:  
        file1.write(data[0])
        file2.write(data[1])
    file1.close()
    file2.close()
    
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

#main
def process(input):
    for index, e in enumerate(input):
        print(f"***\t\tsample {index}\t\t***")
        t=xpuzzle.XPuzzle(2, 4, input[index]) 
        s=search.Search(t)

        p = s.bestFirst()
        print(p[0])
        exportData("ucs", index, p)

        for heuristic in range(2):
            p = s.greedyBFS(heuristic)
            print(p[0])
            exportData("gbfs", index, p, heuristic)

            p  = s.aStar(heuristic)
            print(p[0])
            exportData("astar", index, p, heuristic)

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
# inputData[4] no solution
inputData = importData('samplePuzzles.txt')
print(inputData, end='\n\n')

index = 3
t=xpuzzle.XPuzzle(2, 4, inputData[index]) 
t.printPuzzle()
s=search.Search(t)

print("A*")
for heuristic in range(2):
    p  = s.aStar(heuristic)
    print(p[0])
    exportData("astar", index, p, heuristic)

# process driver
inputData = importData('samplePuzzles.txt')
print(inputData, end='\n\n')
process(inputData)


