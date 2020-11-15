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
#distance = (2-i)+(2-j) if index == -1 else abs(i-(index/self.state.size))+abs(j-(index%self.state.size))
# file.write("\ncost: " + str(moves))
# file.write("\nrunning_time: " + format(time, '.8f'))

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

#sample
def generateInput(total, length=8):
    input = []
    for i in range(total):
        input.append(list(np.random.permutation(length)))
    return input

#analysis
def analysis():
    N = 50
    input = generateInput(N)
    ucs_stat = np.zeros(5)
    gbfs0_stat = np.zeros(5)
    gbfs1_stat = np.zeros(5)
    star0_stat = np.zeros(5)
    star1_stat = np.zeros(5)
    for index, e in enumerate(input):
        print(f"***\t\tsample {index}\t\t***")
        t=xpuzzle.XPuzzle(2, 4, input[index]) 
        s=search.Search(t, 15)

        p = s.uniformCost()
        print(p[0])
        if isinstance(p[0], node.Node):
            ucs_stat+=np.array([len(p[0].solution), len(p[1]), p[0].gn, p[2], 1/N])
        p = s.greedyBFS(0)
        if isinstance(p[0], node.Node):
            gbfs0_stat+=np.array([len(p[0].solution), len(p[1]), p[0].gn, p[2], 1/N])
        p = s.greedyBFS(1)
        if isinstance(p[0], node.Node):
            gbfs1_stat+=np.array([len(p[0].solution), len(p[1]), p[0].gn, p[2], 1/N])
        p  = s.aStar(0)
        if isinstance(p[0], node.Node):
            star0_stat+=np.array([len(p[0].solution), len(p[1]), p[0].gn, p[2], 1/N])
        p  = s.aStar(1)
        if isinstance(p[0], node.Node):
            star1_stat+=np.array([len(p[0].solution), len(p[1]), p[0].gn, p[2], 1/N])
    return np.array([ucs_stat, gbfs0_stat, gbfs1_stat, star0_stat, star1_stat])

#scaleup
def scaleup(row, col):
    N = 1
    star1_stat = []
    for r in range(1, row):
        for c in range(3, col):
            print(f"***\t\tsample [{r}, {c}]\t\t***")
            input = generateInput(N, r*c)
            t=xpuzzle.XPuzzle(r, c, input[0]) 
            s=search.Search(t, 120)

            p  = s.aStar(1)
            print(f"{p[0]} time: {p[2]}")
            if isinstance(p[0], node.Node):
                star1_stat.append([r, c, p[2]])
            else:
                star1_stat.append([r, c, -1])   
    return star1_stat

#main
def process(input):
    for index, e in enumerate(input):
        print(f"***\t\tsample {index}\t\t***")
        t=xpuzzle.XPuzzle(2, 4, input[index]) 
        s=search.Search(t)

        p = s.uniformCost()
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

# In[ ]:
# process driver
inputData = importData('samplePuzzles.txt')
print(inputData, end='\n\n')
process(inputData)

# In[ ]:
# analysis
stat1 = analysis()
print(stat1)

# In[ ]:
# scaleup
stat2 = scaleup(6, 8)
print(stat2)

