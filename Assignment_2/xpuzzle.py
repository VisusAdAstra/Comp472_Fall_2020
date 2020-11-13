import sys
from random import randint


class XPuzzle:
	'''
    Instantiates the class, defining the start node
    '''
	def __init__(self, row, col, list):
		self.row=row
		self.col=col
		self.puzzle=[]
		self.zero=(0,0)
		self.moves=["U","D","L","R","W","C"]
		self.costs=[1, 1, 1, 1, 2, 3]
		for i in range(0, self.row):
			self.puzzle.append([])
			for j in range(0, self.col):
				self.puzzle[i].append(int(list[i*self.col+j]))
				if int(list[i*self.col+j])==0:
					self.zero=(i,j)

	def initialize(self, list):
		for i in range(0,self.row):
			for j in range(0,self.col):
				if int(list[i*self.col+j])==0:
					self.zero=(i,j)
				self.puzzle[i][j]=int(list[i*self.col+j])

	def readPuzzle(self, string):
		a=string.split(" ")
		count=0
		for i in range(0,self.row):
			for j in range(0,self.col):
				if int(a[count])==0:
					self.zero=(i,j)
				self.puzzle[i][j]=int(a[count])
				count+=1

	# 2 condition for 2 goal state
	def checkPuzzle(self):
		count=1
		correct = True
		for i in range(0, self.row):
			for j in range(0, self.col):
				if self.puzzle[i][j]!=(count%(self.row*self.col)):
					correct = False
				count+=1
		if correct == True:
			return True
		count=1
		for j in range(0, self.col):
			for i in range(0, self.row):
				if self.puzzle[i][j]!=(count%(self.row*self.col)):
					return False
				count+=1
		return True

	# def swap(self,(x1,y1),(x2,y2)):#sublist parameters are not supported in 3.x
	def swap(self,p1,p2):
		x1, y1 = p1
		x2, y2 = p2
		temp=self.puzzle[x1][y1]
		self.puzzle[x1][y1]=self.puzzle[x2][y2]
		self.puzzle[x2][y2]=temp

	def up(self):
		if (self.zero[0]!=0):
			self.swap((self.zero[0]-1,self.zero[1]),self.zero)
			self.zero=(self.zero[0]-1,self.zero[1])
			return 1
		else:
			self.swap((self.row-1,self.zero[1]),self.zero)
			self.zero=(self.row-1,self.zero[1])
			return 2

	def down(self):
		if (self.zero[0]!=self.row-1):
			self.swap((self.zero[0]+1,self.zero[1]),self.zero)
			self.zero=(self.zero[0]+1,self.zero[1])
			return 1
		else:
			self.swap((0,self.zero[1]),self.zero)
			self.zero=(0,self.zero[1])
			return 2

	def left(self):
		if (self.zero[1]!=0):
			self.swap((self.zero[0],self.zero[1]-1),self.zero)
			self.zero=(self.zero[0],self.zero[1]-1)
			return 1
		else:
			self.swap((self.zero[0],self.col-1),self.zero)
			self.zero=(self.zero[0],self.col-1)
			return 2

	def right(self):
		if (self.zero[1]!=self.col-1):
			self.swap((self.zero[0],self.zero[1]+1),self.zero)
			self.zero=(self.zero[0],self.zero[1]+1)
			return 1
		else:
			self.swap((self.zero[0],0),self.zero)
			self.zero=(self.zero[0],0)
			return 2

	def corner(self):
		x = abs(self.row-1 - self.zero[0])
		y = abs(self.col-1 - self.zero[1])
		self.swap((x,y),self.zero)
		self.zero=(x,y)
		return 3
	
	def printPuzzle(self):
		for i in range(0,self.row):
			for j in range(0,self.col):
				print(self.puzzle[i][j], end=" ")
			print("")
		print("")

	def doMove(self,move):
		if move=="U":
			self.up()
		if move=="D":
			self.down()
		if move=="L":
			self.left()
		if move=="R":
			self.right()
		if move=="C":
			if self.zero == (0, 0) or self.zero == (0, self.col-1) or self.zero == (self.row-1, 0) or self.zero == (self.row-1, self.col-1):
				self.corner()
	
