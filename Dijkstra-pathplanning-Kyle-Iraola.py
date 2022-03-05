import numpy as np
from queue import PriorityQueue
from collections import deque
import cv2
import math
import os
import sys
from uuid import uuid4 #Library to create unique strings for indexing
#########################################################################
#Input function definitions
def getStartX():
    xBound = range(6,395)
    startX = int(input("Enter starting X coordinate:"))
    while True:
        if startX in xBound:
            break
        else:
            startX = int(input("OUT OF BOUNDS! Please choose an X between 6 and 394.\nEnter starting X coordinate:"))
            continue
    return startX

def getStartY():
    yBound = range(6,245)
    startY = int(input("Enter starting Y coordinate:"))
    while True:
        if startY in yBound:
            break
        else:
            startY = int(input("OUT OF BOUNDS! Please choose a Y between 6 and 244.\nEnter starting X coordinate:"))
            continue
    return startY

def getGoalX():
    xBound = range(6,395)
    goalX = int(input("Enter goal X coordinate:"))
    while True:
        if goalX in xBound:
            break
        else:
            goalX = int(input("OUT OF BOUNDS! Please choose an X between 6 and 394.\nEnter starting X coordinate:"))
            continue
    return goalX

def getGoalY():
    yBound = range(6,245)
    goalY = int(input("Enter goal Y coordinate:"))
    while True:
        if goalY in yBound:
            break
        else:
            goalY = int(input("OUT OF BOUNDS! Please choose a Y between 6 and 244.\nEnter starting Y coordinate:"))
            continue
    return goalY


#Function definitions

def moveUp(node): #Move action (0,-1), action Cost: 1
    x = node[0]
    y = node[1]
    moveUpNode = (x, y - 1)
    return moveUpNode, 1

def moveUpRight(node): #Move action (1,-1), action Cost: math.sqrt(2)
    x = node[0]
    y = node[1]
    moveUpRightNode = (x + 1, y - 1)
    return moveUpRightNode, math.sqrt(2)

def moveRight(node): #Move action (1,0), action Cost: 1
    x = node[0]
    y = node[1]
    moveRightNode = (x + 1, y)
    return moveRightNode, 1

def moveDownRight(node): #Move action (1,1), action Cost: math.sqrt(2)
    x = node[0]
    y = node[1]
    moveUpRightNode = (x + 1, y + 1)
    return moveUpRightNode, math.sqrt(2)


def moveDown(node): #Move action (0,1), action Cost: 1
    x = node[0]
    y = node[1]
    moveDownNode = (x, y + 1)
    return moveDownNode, 1

def moveDownLeft(node): #Move action (-1,1), action Cost: math.sqrt(2)
    x = node[0]
    y = node[1]
    moveDownLeftNode = (x - 1, y + 1)
    return moveDownLeftNode, math.sqrt(2)

def moveLeft(node): #Move action (-1,0), action Cost: 1
    x = node[0]
    y = node[1]
    moveLeftNode = (x - 1, y)
    return moveLeftNode, 1

def moveUpLeft(node): #Move action (-1,-1), action Cost: math.sqrt(2)
    x = node[0]
    y = node[1]
    moveUpLeftNode = (x - 1, y - 1)
    return moveUpLeftNode, math.sqrt(2)

#Takes in information and puts the children of the current node in the open list
def getChildren(environment, C2Cmatrix, openList, closedList, currentNodeInfo):
    #Node info: (Cost, index, parent index, node coordinate)
    currentC2C = currentNodeInfo[0] #Current node cost -> currentC2C
    parentIndex = currentNodeInfo[1] #Current node index -> current index
    currentCoord = currentNodeInfo[3] #Current node coordinate

    children = []

    actionSet = [moveUp, moveUpRight, moveRight, moveDownRight, moveDown,
                 moveDownLeft, moveLeft, moveUpLeft]
    for action in actionSet:
        coordinate, cost = action(currentCoord)
        if environment[coordinate[1], coordinate[0]] == 0:
            infoShell = (round((currentC2C + cost),4), str(uuid4()), parentIndex, coordinate)
            children.append(infoShell)
    return children

def isChildInOpenList(coordinate, openListQueue):
    for i in range(len(openListQueue)):
        if openListQueue[i][3] == coordinate:
            return True, i
    return False, None

#Main Dijkstra algorithm pathfinding
def Dijkstra(environment, animation, C2Cmatrix, openList, closedList, startNode, goalNode):
    while len(openList.queue) != 0:
        currentNodeInfo = openList.get()
        closedList.append(currentNodeInfo)
        animation[currentNodeInfo[3][1], currentNodeInfo[3][0]] = .5
        cv2.imshow("Map", animation)
        cv2.waitKey(1)
        if currentNodeInfo[3] == goalNode:
            filename = "map.jpg"
            animation = animation*255
            print(animation)
            cv2.imwrite(filename, animation.astype(np.int32))
            return backtrack(currentNodeInfo, closedList, animation, filename)
        else:
            for child in getChildren(environment, C2Cmatrix, openList, closedList, currentNodeInfo):
                cost, index, parent, coordinate = child
                if child not in closedList:
                    exists, index = isChildInOpenList(coordinate, openList.queue)
                    if not exists and C2Cmatrix[coordinate[1], coordinate[0]] == float("inf"):
                        C2Cmatrix[coordinate[1], coordinate[0]] = cost
                        openList.put(child)
                    else:
                        if C2Cmatrix[coordinate[1], coordinate[0]] > cost and exists:
                            C2Cmatrix[coordinate[1], coordinate[0]] = cost
                            openList.queue[index] = child
    return False

#Backtracking function that looks through the final closed list and iterates through. Outputs the coordinate path as a list [Start -> Goal]
def backtrack(currentNodeInfo, closedList, animation, filename):
    print(str(os.getcwd()) + "\\" + filename)
    img = cv2.imread((str(os.getcwd()) + "\\" + filename), cv2.IMREAD_COLOR)
    path = deque([])
    while currentNodeInfo[3] != startNode:
        path.appendleft(currentNodeInfo[3])
        img[currentNodeInfo[3][1], currentNodeInfo[3][0]] = [0,0,255] # Make it colored
        cv2.imshow("Map", img.astype(np.uint8))
        cv2.waitKey(1)
        for i in range(len(closedList)):
            if closedList[i][1] == currentNodeInfo[2]:
                currentNodeInfo = closedList[i]
    path.appendleft(startNode)
    cv2.waitKey(1000)
    return list(path)


#########################################################################
#Creating an array to iterate over for map creation
environment = np.zeros((250,400))

#Geometric obstacle definitions. Half-plane equations calculated with MATLAB, 5 decimal  precision
circle = lambda x, y: ((x - 300)**2 + (y - 65)**2 - 45**2) <= 0
hexagon = lambda x, y: 0 >= (-.577*x + 219.28-y) and 0 >= (.577*x - 11.658-y) and 0 <= (240-x) and 0 < (-.557*x + 311.66-y) and 0 <= (.577*x + 80.718-y) and 0 >= (160-x)
polygonUpper = lambda x, y: 0 >= (-.316*x + 71.148-y) and 0 <= (-.714*x + 128.29-y) and 0 <= (.0766*x + 60.405 -y)
polygonLower = lambda x, y: 0 >= (3.263*x - 213.15-y) and 0 <= (1.182*x + 30.176-y) and 0 >= (.0766*x + 60.405 -y)
mapBounds = lambda x, y: 0 < (5-x) or 0 > (394-x) or 0 < (5-y) or 0 > (244-y)

for y in range(0,250):
    for x in range(0,400):
        if circle(x,y):
            environment[y,x] = 1
        elif hexagon(x,y):
            environment[y,x] = 1
        elif polygonUpper(x,y):
            environment[y,x] = 1
        elif polygonLower(x,y):
            environment[y,x] = 1
        elif mapBounds(x,y):
            environment[y,x] = 1
        else:
            environment[y,x] = 0

#Creating a new matrix to change values and update the visulaization to while keeping the original matrix pure for Dijkstra
animation = environment.copy()

#########################################################################
#Creating a matrix representing the cost 2 come from the start node
C2Cmatrix = environment.copy()
C2Cmatrix = np.where(C2Cmatrix == 0, float("inf"), C2Cmatrix)
C2Cmatrix = np.where(C2Cmatrix == 1, -1, C2Cmatrix)

#########################################################################
#Getting user inputs for starting and goal nodes. Only accepts "good" information

startX = getStartX()
startY = getStartY()
goalX = getGoalX()
goalY = getGoalY()

#Check whether either inputed node are in obstacle space, and if so promt user to try again.
if environment[startY,startX] == 1:
    sys.exit("Chosen Start Node is in obstacle space. Please try again!")
elif environment[goalY,goalX] == 1:
    sys.exit("Chosen Goal Node is in obstacle space. Please try again!")
elif startX == goalX and startY == goalY:
    sys.exit("Chosen Start and Goal nodes are identical. Please try again!")
else:
    startNode = (startX, startY)
    goalNode = (goalX, goalY)

        
########################################################################
#Initializing start node's info as (cost2come, index, parent index, coordinate tuple)
startNodeInfo = (0, str(uuid4()), str(uuid4()), startNode)
C2Cmatrix[startNode[1], startNode[0]] = 0
openList = PriorityQueue()

#Put the start node inside the open list first

openList.put(startNodeInfo)

#Initializing the closed list
closedList = []

#Perform Dijkstra. It will print out backtracking list and also show an OpenCV animation on-screen during the search.
print(Dijkstra(environment, animation, C2Cmatrix, openList, closedList, startNode, goalNode))
print("\nPress any key to exit the window!")
cv2.waitKey(0)
cv2.destroyAllWindows()


