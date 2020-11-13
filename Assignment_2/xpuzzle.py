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
	
	def permute(self,numPerm):
		for i in range(0,numPerm):
			self.doMove(self.moves[randint(0,3)])
	
	def parseMoveSequence(self,string):
		for m in string:
			self.doMove(m)

	def manhattanDistance(self):
		result = [0, 0]
		distance = [0, 0]
		for i in range(0,self.row):
			for j in range(0,self.col):
				index = self.puzzle[i][j] - 1
				if index == -1: 
					distance[0] = (self.row-1-i)+(self.col-1-j)
					distance[1] = (self.row-1-i)+(self.col-1-j)
				else:
					distance[0] = abs(i-(index//self.col)) + abs(j-(index%self.col))
					distance[1] = abs(i-(index%self.row)) + abs(j-(index//self.row))
				result[0] += distance[0]
				result[1] += distance[1]
		print(result)
		if result[0] < result[1]:
			return result[0]
		else:
			return result[1]
			
		

#t=tilePuzzle(int(sys.argv[1]))
#t.permute(int(sys.argv[2]))
#t.printPuzzle()
  
# Module Methods

EIGHT_PUZZLE_DATA = [[1, 0, 2, 3, 4, 5, 6, 7, 8], 
                     [1, 7, 8, 2, 3, 4, 5, 6, 0], 
                     [4, 3, 2, 7, 0, 5, 1, 6, 8], 
                     [5, 1, 3, 4, 0, 2, 6, 7, 8], 
                     [1, 2, 5, 7, 6, 8, 0, 4, 3], 
                     [0, 3, 1, 6, 8, 2, 7, 5, 4]]

def loadEightPuzzle(puzzleNumber):
  """
    puzzleNumber: The number of the eight puzzle to load.
    
    Returns an eight puzzle object generated from one of the
    provided puzzles in EIGHT_PUZZLE_DATA.
    
    puzzleNumber can range from 0 to 5.
    
    >>> print loadEightPuzzle(0)
    -------------
    | 1 |   | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------
    | 6 | 7 | 8 |
    -------------
  """
  return EightPuzzleState(EIGHT_PUZZLE_DATA[puzzleNumber])

def createRandomEightPuzzle(moves=100):
 """
   moves: number of random moves to apply

   Creates a random eight puzzle by applying
   a series of 'moves' random moves to a solved
   puzzle.
 """
 puzzle = EightPuzzleState([0,1,2,3,4,5,6,7,8])
 for i in range(moves):
   # Execute a random legal move
   puzzle = puzzle.result(random.sample(puzzle.legalMoves(), 1)[0])
 return puzzle
