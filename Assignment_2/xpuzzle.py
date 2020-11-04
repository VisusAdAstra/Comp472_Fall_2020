import sys
from random import randint


class XPuzzle:
	def __init__(self,size):
		self.size=size
		self.puzzle=[]
		self.zero=(0,0)
		self.moves=["U","D","L","R"]
		count=1
		for i in range(0,size):
			self.puzzle.append([])
			for j in range(0,size):
				self.puzzle[i].append(count)
				count+=1
		self.puzzle[size-1][size-1]=0
		self.zero=(size-1,size-1)

	def readPuzzle(self,string):
		a=string.split(" ")
		count=0
		for i in range(0,self.size):
			for j in range(0,self.size):
				if int(a[count])==0:
					self.zero=(i,j)
				self.puzzle[i][j]=int(a[count])
				count+=1

	def checkPuzzle(self):
		count=1
		for i in range(0,self.size):
			for j in range(0,self.size):
				if self.puzzle[i][j]!=(count%(self.size*self.size)):
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
		if (self.zero[0]!=self.size-1):
			self.swap((self.zero[0]+1,self.zero[1]),self.zero)
			self.zero=(self.zero[0]+1,self.zero[1])

	def left(self):
		if (self.zero[1]!=0):
			self.swap((self.zero[0],self.zero[1]-1),self.zero)
			self.zero=(self.zero[0],self.zero[1]-1)


	def right(self):
		if (self.zero[1]!=self.size-1):
			self.swap((self.zero[0],self.zero[1]+1),self.zero)
			self.zero=(self.zero[0],self.zero[1]+1)
	
	def printPuzzle(self):
		for i in range(0,self.size):
			for j in range(0,self.size):
				print(self.puzzle[i][j], end=" ")
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
	
	def permute(self,numPerm):
		for i in range(0,numPerm):
			self.doMove(self.moves[randint(0,3)])
	
	def parseMoveSequence(self,string):
		for m in string:
			self.doMove(m)
			
		

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
