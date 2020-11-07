import sys
from random import randint


class XPuzzle:
	'''
    Instantiates the class, defining the start node
    '''
	def __init__(self, row, col):
		self.row=row
		self.col=col
		self.puzzle=[]
		self.zero=(0,0)
		self.moves=["U","D","L","R","WM","DM"]
		count=1
		for i in range(0,row):
			self.puzzle.append([])
			for j in range(0,col):
				self.puzzle[i].append(count)
				count+=1
		self.puzzle[row-1][col-1]=0
		self.zero=(row-1,col-1)

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

	def checkPuzzle(self):
		count=1
		for i in range(0,self.row):
			for j in range(0,self.col):
				if self.puzzle[i][j]!=(count%(self.row*self.col)):
					return False
				count+=1
		return True

	
	#def swap(self,(x1,y1),(x2,y2)):#sublist parameters are not supported in 3.x
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

	def down(self):
		if (self.zero[0]!=self.row-1):
			self.swap((self.zero[0]+1,self.zero[1]),self.zero)
			self.zero=(self.zero[0]+1,self.zero[1])

	def left(self):
		if (self.zero[1]!=0):
			self.swap((self.zero[0],self.zero[1]-1),self.zero)
			self.zero=(self.zero[0],self.zero[1]-1)


	def right(self):
		if (self.zero[1]!=self.col-1):
			self.swap((self.zero[0],self.zero[1]+1),self.zero)
			self.zero=(self.zero[0],self.zero[1]+1)
	
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
	
