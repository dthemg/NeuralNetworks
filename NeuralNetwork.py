# -*- coding: utf-8 -*-
'''

Neural Networks - figures learn to find green balls and avoid purple balls

Try playing around with the three overhead constants first defined:
    KEYBOARD: Creates a single eater that is moved around with by using keyboard inputs
    INFINITE_DUARATION: Evolving algorithm is turned off so no generation-shift occurs
    SHOW_BRAIN: Shows the activity within a eater on the map.

David Montgomery 21/02/2017

'''

import numpy as np
import pygame as pg
import time

# Define overhead constants
KEYBOARD = False
INFINITE_DURATION = False
SHOW_BRAIN = True

# Define neural networks constants
VISUALIZATIONS = True
NUMBER_OF_GENERATIONS = 200
NUMBER_OF_NODES = 3
NUMBER_OF_INPUTS = 12
NUMBER_OF_OUTPUTS = 3
START_SIGMA = 0.5
PUNISHMENT = 2
WITH_MUTATION = True
MUTATE_PROBABILITY = 0.05
import matplotlib.pyplot as plt


# Define eater constants
DIRECTION_INCREMENT = 4
ACCELERATION = 0.5
DEACCELERATION = 1
MAX_SPEED = 6
FIGURE_RADIUS = 8
FIGURE_LINE_THICKNESS = 2
OBSERVATION_LINE_THICKNESS = 1
OBSERVATION_LINE_LENGTH = 150

# Define game constants
RESET_TIME = 15
GAME_TIME = 2*RESET_TIME
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700
NUMBER_OF_EATERS = 5
NUMBER_OF_FOOD = 200
NUMBER_OF_POISON = 50
FOOD_RADIUS = 8
SIDE_LIMIT = 10
EATER_OBSERVE_EXTRA = 10

# Define normation constants
MIN_WALL_DISTANCE = 0
MAX_WALL_DISTANCE = 160
MIN_FOOD_DISTANCE = 0
MAX_FOOD_DISTANCE = 135

# Define brain constants
BRAIN_WIDTH = 600
SIDE_LINE_THICKNESS = 5
MARGIN_TOP = 50
MARGIN_SIDE = 50
MARGIN_INPUT_BOXES = 10
INPUTBOX_SIZE = 40
NODE_RADIUS = 20
OUTPUTBOX_SIZE = 80
BOX_WIDTH = BRAIN_WIDTH - MARGIN_SIDE*2
BOX_HEIGHT = SCREEN_HEIGHT - MARGIN_TOP*2
BOX_X_START = SCREEN_WIDTH + MARGIN_SIDE
BOX_X_END = SCREEN_WIDTH + BRAIN_WIDTH - MARGIN_SIDE
BOX_Y_START = MARGIN_TOP
BOX_Y_END = SCREEN_HEIGHT - MARGIN_TOP
INPUT_MARGIN = int(round((BOX_HEIGHT - NUMBER_OF_INPUTS*INPUTBOX_SIZE)/(NUMBER_OF_INPUTS - 1)))
NODE_DIST = int(round(BOX_HEIGHT/(NUMBER_OF_NODES-1)))
OUTPUT_MARGIN = 150
OUTPUTBOX_MARGIN = int(round((BOX_HEIGHT- 2*OUTPUT_MARGIN - NUMBER_OF_OUTPUTS*OUTPUTBOX_SIZE)/(NUMBER_OF_OUTPUTS - 1)))

# Define colors used
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (50,230,50)
BLUE = (0,0,100)
LIGHT_BLUE = (0,0,170)
LIGHT_GREEN = (50,170,50)
LIGHT_GRAY = (150,150,150)
PINKY_PURPLE = (204,0,102)
TURQUOISE = (0,153,153)
BROWN = (153,76,0)
PURPLE = (153, 0, 153)

NORMAL_OBSERVATION_LINE_COLOR = LIGHT_BLUE
SUPER_OBSERVATION_LINE_COLOR = LIGHT_GREEN
FIGURE_COLOR = BLUE
FIGURE_LINE_COLOR = RED
FOOD_COLOR = GREEN
BACKGROUND = LIGHT_GRAY
POISON_COLOR = PURPLE

SEARCH_LIM = FOOD_RADIUS + OBSERVATION_LINE_LENGTH

if INFINITE_DURATION:
    RESET_TIME = 3600
    GAME_TIME = 7200

# Drawing class, handles all drawings
class Drawer(object):
    
    def __init__(self, screen):
        self.screen = screen
        self.fontGenText = pg.font.SysFont(None, 50)
        self.fontDirText = pg.font.SysFont(None, 25)

    def setNewBrainColors(self):
        W1 = self.brainEater.NeuralNetwork.W1 
        W2 = self.brainEater.NeuralNetwork.W2
        for i in range(NUMBER_OF_NODES):
            for k in range(NUMBER_OF_INPUTS):
                self.colorListW1[i][k][:] = self._brainColor(sigmoid(W1[i][k]))
        for i in range(NUMBER_OF_OUTPUTS):
            for k in range(NUMBER_OF_NODES):
                self.colorListW2[i][k][:] = self._brainColor(sigmoid(W2[i][k]))
                
    """3-dimensional matrices to store color values, probably a really dumb way to do this"""
    def setBrainEater(self, firstEater):
        self.brainEater = firstEater
        self.colorListW1 = np.empty((NUMBER_OF_NODES, NUMBER_OF_INPUTS, 3))
        self.colorListW2 = np.empty((NUMBER_OF_OUTPUTS, NUMBER_OF_NODES, 3))

    """Input colors"""
    def _inputColor(self, value):
        if value < 0.01:
            redVal = 255
            greenVal = 0
        else:
            redVal = 0
            greenVal = int(min(round(256*value), 255))
        return (redVal, greenVal, 0)

    """Converts a value into a color"""
    def _brainColor(self, value):
        if value < 0.5:
            redVal = int(max(round(220-440*value), 40))
            greenVal = 0
        else:
            redVal = 0
            greenVal = int(min(round(440*value-180), 255))
        return (redVal, greenVal, 0)
    
    """Draws the brain structure for the designated eater"""
    def drawBrain(self):
        # Gather all necessary data for all colors
        inputVector, a2, a3, resultMoves = self.brainEater.NeuralNetwork.propagate(self.brainEater.getObservationLines(), allInfo=True)
        pg.draw.line(self.screen, BLACK, [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], SIDE_LINE_THICKNESS)
        pg.draw.circle(self.screen, RED, [int(round(self.brainEater.xPos)), int(round(self.brainEater.yPos))], 3*FIGURE_RADIUS, 2)
        # Drawing first row of boxes:
        xStart = BOX_X_START
        xLines = xStart + INPUTBOX_SIZE        
        for k in range(NUMBER_OF_INPUTS):
            color = self._inputColor(inputVector[k])
            yStart = BOX_Y_START + k*(INPUTBOX_SIZE + INPUT_MARGIN)
            pg.draw.polygon(self.screen, color, [(xStart, yStart), (xStart + INPUTBOX_SIZE, yStart), \
                                    (xStart + INPUTBOX_SIZE, yStart + INPUTBOX_SIZE), (xStart, yStart + INPUTBOX_SIZE)])
            yLines = BOX_Y_START + k*(INPUTBOX_SIZE + INPUT_MARGIN) + int(round(INPUTBOX_SIZE/2))       
            xEnd = int(round(BOX_X_START + BOX_WIDTH/2 - NODE_RADIUS))
            for i in range(NUMBER_OF_NODES):
                color = tuple(self.colorListW1[i][k][:])
                pg.draw.line(self.screen, color, [xLines, yLines], \
                                [xEnd, BOX_Y_START + i*(NODE_DIST)], 1)
        # Drawing center node circles    
        xStart = int(round(BOX_X_START + BOX_WIDTH/2))
        xLines = xStart + NODE_RADIUS
        for k in range(NUMBER_OF_NODES):
            color = self._brainColor(a2[k])
            pg.draw.circle(self.screen, color, [xStart,BOX_Y_START + k*(NODE_DIST)], NODE_RADIUS)     
            yLines = BOX_Y_START + k*(NODE_DIST)
            for i in range(NUMBER_OF_OUTPUTS):
                color = tuple(self.colorListW2[i][k][:])
                pg.draw.line(self.screen, color, [xLines, yLines], \
                                    [BOX_X_END - OUTPUTBOX_SIZE, BOX_Y_START + OUTPUT_MARGIN + \
                                   i*(OUTPUTBOX_SIZE + OUTPUTBOX_MARGIN) + OUTPUTBOX_SIZE/2], 1)
        # Drawing output boxes
        xStart = BOX_X_END - OUTPUTBOX_SIZE
        for k in range(NUMBER_OF_OUTPUTS):
            yStart = BOX_Y_START + OUTPUT_MARGIN + k*(OUTPUTBOX_SIZE + OUTPUTBOX_MARGIN)
            color = self._brainColor(a3[k])
            pg.draw.polygon(self.screen, color, [(xStart, yStart), (xStart + OUTPUTBOX_SIZE, yStart), \
                                    (xStart + OUTPUTBOX_SIZE, yStart + OUTPUTBOX_SIZE), (xStart, yStart + OUTPUTBOX_SIZE)])
            if resultMoves[k]:
                if k == 0:
                    text = self.fontDirText.render("Left", True, BLACK)
                    self.screen.blit(text, (xStart + 23, yStart + 32))    
                elif k == 1:
                    text = self.fontDirText.render("Right", True, BLACK)
                    self.screen.blit(text, (xStart + 17, yStart + 32))    
                elif k == 2:
                    text = self.fontDirText.render("Forward", True, BLACK)
                    self.screen.blit(text, (xStart + 6, yStart + 32))       
    
    def makeGenerationCounter(self, evolver):
        text = self.fontGenText.render("Generation: " + str(evolver.generation), True, BLACK)
        self.screen.blit(text, (round(SCREEN_WIDTH/2 - 100), 40))
    
    def makeScoreCounter(self, totScore):
        font = pg.font.SysFont(None, 25)
        text = font.render("Generation score: " + str(totScore), True, BLACK)
        self.screen.blit(text, (20,45))
        
    def drawFoods(self, foodPositions):
        L = len(foodPositions[0])
        for i in range(L):
            pg.draw.circle(self.screen, FOOD_COLOR, [round(foodPositions[0][i]), round(foodPositions[1][i])], FOOD_RADIUS)
        
    def drawPoison(self, poisonPositions):
        L = len(poisonPositions[0])
        for i in range(L):
            pg.draw.circle(self.screen, POISON_COLOR, [round(poisonPositions[0][i]), round(poisonPositions[1][i])], FOOD_RADIUS)
    
    def drawEater(self, eater):
        obsLines = eater.getObservationLines()
        xCoord = int(round(eater.xPos))
        yCoord = int(round(eater.yPos))
        pg.draw.circle(self.screen, eater.color, [xCoord, yCoord], FIGURE_RADIUS)
        
        # Draw observation lines
        angles = [-30, -15, 0, 15, 30]
        for i in range(len(angles)):
            # Decide on right color for center line
            if i == 2:
                if obsLines[5] == 0:
                    col = SUPER_OBSERVATION_LINE_COLOR
                else:
                    col = (round(min(255, 100 + 150*obsLines[5])), round(min(255, 100 + 150*obsLines[5])), 0)

            # If food is closer than poison
            elif obsLines[i] > obsLines[i+6] and obsLines[i] > 0:
                col = (round(min(255, 100 + 140*obsLines[i])), 0, 0)
            # If poison is closer than food
            elif obsLines[i] <= obsLines[i+6] and obsLines[i+6] > 0:
                col = (round(min(255, 153 + 70*obsLines[i+6])), round(min(255, 76 + 30*obsLines[i+6])), 0)
            else:
                col = NORMAL_OBSERVATION_LINE_COLOR
            # Draw the lines
            angle = angles[i]
            try:
                pg.draw.line(self.screen, col, [xCoord, yCoord], \
                        [round(eater.xPos + OBSERVATION_LINE_LENGTH*np.cos(degToRad(eater.direction + angle))), \
                        round(eater.yPos + OBSERVATION_LINE_LENGTH*np.sin(degToRad(eater.direction + angle)))], OBSERVATION_LINE_THICKNESS)
            except TypeError:
                print(col)
        
        # Draw direction line
        pg.draw.line(self.screen, FIGURE_LINE_COLOR, [xCoord, yCoord], 
                     [round(eater.xPos + 1.5*FIGURE_RADIUS*np.cos(degToRad(eater.direction))), 
                     round(eater.yPos + 1.5*FIGURE_RADIUS*np.sin(degToRad(eater.direction)))], FIGURE_LINE_THICKNESS) 
        

# Projects vector v onto vector s with s of unitary length
def projection(v, s):
    p = np.dot(v, s)
    u = np.multiply(p, s)
    return p, u
    
# Converts degrees to radians
def degToRad(deg):
    return deg*np.pi/180

# Activation function for neural network
def sigmoid(x):
    return 1/(1+np.exp(-x))

    
"""Creates and stores positions of food, also detects food, and now also poison"""
class FoodPoisonHandler(object):
    
    def __init__(self):
        self.foodPositions = [[],[]]
        self.poisonPositions = [[],[]]
        self.updateFoodsList()
        self.updatePoisonList()
        self.angles = [-30, -15, 0, 15, 30]

    """Resets entire foodsList"""
    def resetList(self):
        self.foodPositions = [[],[]]
        
    def _updateList(self, length, updateList):
        numberOfNewItems = length - len(updateList[0])
        xNew = np.random.uniform(low=20, high=SCREEN_WIDTH-20, size=numberOfNewItems)
        xList = np.ndarray.tolist(xNew)
        updateList[0].extend(xList)
        yNew = np.random.uniform(low=20, high=SCREEN_HEIGHT-20, size=numberOfNewItems)
        yList = np.ndarray.tolist(yNew)
        updateList[1].extend(yList)
        return updateList
    
    '''Keeps number of foods constant'''
    def updateFoodsList(self):
        self.foodPositions =  self._updateList(NUMBER_OF_FOOD, self.foodPositions)
        
    def updatePoisonList(self):
        self.poisonPositions = self._updateList(NUMBER_OF_POISON, self.poisonPositions)
    
    """p1 on form [x1, y1]"""
    def _detectIfFound(self, p1, p2):
        if np.linalg.norm([p1[0] - p2[0], p2[1] - p1[1]]) < FIGURE_RADIUS + FOOD_RADIUS:
            return True
        else:
            return False
    
    def _nearby(self, eater, searchList):
        nearby = []
        for i in range(len(searchList[0])):
            if abs(searchList[0][i] - eater.xPos) < SEARCH_LIM and \
                    abs(searchList[1][i] - eater.yPos) < SEARCH_LIM:
                nearby.append(i)
        return nearby
    
    def _detectList(self, eater, searchList, foods):
        foundList = []
        nearby = self._nearby(eater, searchList)
        if nearby:
            for i in nearby:
                if self._detectIfFound([searchList[0][i], searchList[1][i]], \
                                        [eater.xPos, eater.yPos]):
                    foundList.append(i)
                    if foods:
                        eater.increaseScore()
                    if not foods:
                        eater.decreaseScore()
            for index in sorted(foundList, reverse=True):
                del searchList[0][index]
                del searchList[1][index]
        return searchList
    
    """Removes index of found food, gives eater score for finding food"""    
    def detectFoodEaten(self, eater):
        self.foodPositions = self._detectList(eater, self.foodPositions, True)
    
    """Removes index of found poison, decreases eater score for finding poison"""   
    def detectPoisonFound(self, eater):
        self.poisonPositions = self._detectList(eater, self.poisonPositions, False)

    """Sees if number of foods has decreased"""
    def newFoodsRequired(self):
        if NUMBER_OF_FOOD > len(self.foodPositions[0]):
            return True
        else:
            return False

    def newPoisonRequired(self):
        if NUMBER_OF_POISON > len(self.poisonPositions[0]):
            return True
        else:
            return False
    
    def _findDistance(self, eater, searchList):
        nearby = self._nearby(eater, searchList)
        angles = [i + eater.direction for i in self.angles]
        x_vec = [0]*len(angles)
        for k in range(len(angles)):
            angle = angles[k]
            s = np.array([np.cos(degToRad(angle)), np.sin(degToRad(angle))])
            for i in nearby:
                v = np.array([searchList[0][i] - eater.xPos, searchList[1][i]  - eater.yPos])
                if np.linalg.norm(v) < OBSERVATION_LINE_LENGTH:
                    [p, u] = projection(v, s)
                    if p > 0 and np.linalg.norm(v - u) < FIGURE_RADIUS:
                        if x_vec[k] == 0:
                            x_vec[k] = p
                        else:
                            if x_vec[k] > p:
                                x_vec[k] = p
        return x_vec
        
    def findFoodDistances(self, eater):
        return self._findDistance(eater, self.foodPositions)
    
    def findPoisonDistances(self, eater):
        return self._findDistance(eater, self.poisonPositions)
        
    def getFoodPositions(self):
        return self.foodPositions
        
    def getPoisonPositions(self):
        return self.poisonPositions
        
class Eater(object):
    def __init__(self):
        self.newPosition()
        self.direction = np.random.uniform(low=0, high=360)
        self.speed = 0      
        self.score = 0
        self.observationLines = np.zeros(NUMBER_OF_INPUTS)
        # Set final term of observationLines as constant max
        self.observationLines[11] = 1
        self.NeuralNetwork = NeuralNetwork()
        self.color = BLUE
        
    
    def resetColor(self):
        self.color = BLUE
    
    def decreaseScore(self):
        self.score -= PUNISHMENT
    
    def setColor(self, parent, child, mutant):
        if parent and not mutant:
            self.color = TURQUOISE
        if child and not mutant:
            self.color = PINKY_PURPLE
        if mutant:
            self.color = BROWN
        
    def resetScore(self):
        self.score = 0
        
    def increaseScore(self):
        self.score += 1
        
    def getScore(self):
        return self.score
        
    def newPosition(self):
        self.xPos = np.random.uniform(low=0, high=SCREEN_WIDTH)
        self.yPos = np.random.uniform(low=0, high=SCREEN_HEIGHT)

    """Updates Eater's position, direction and speed. moveVector 0's and 1's"""
    def eaterMove(self, moveVector):
        self.direction = self.direction +  DIRECTION_INCREMENT*(moveVector[1] - moveVector[0])
        if self.direction < 0:
            self.direction += 360
        if self.direction > 360:
            self.direction -= 360

        # If eater is out of bounds but on way inwards:
        if self._outOfBounds(self.xPos, self.yPos):
            if not self._outOfBounds(self.xPos + self.speed*np.cos(degToRad(self.direction)), \
                        self.yPos + self.speed*np.sin(degToRad(self.direction))):
                self.xPos += self.speed*np.cos(degToRad(self.direction))
                self.yPos += self.speed*np.sin(degToRad(self.direction))

        else:
            if self.speed < MAX_SPEED and moveVector[2] == 1:
                self.speed += ACCELERATION
            elif self.speed > 0:
                self.speed -= DEACCELERATION
                if self.speed < 0:
                    self.speed = 0
                    
            self.xPos += self.speed*np.cos(degToRad(self.direction))
            self.yPos += self.speed*np.sin(degToRad(self.direction))
         
    """Normalizes wall distance to a value between 0 and 1""" 
    def _normalizeWall(self, incWall):
        return incWall / (MAX_WALL_DISTANCE - MIN_WALL_DISTANCE) + MIN_WALL_DISTANCE
        
    """Normalizes food distance to a value between 0 and 1"""
    def _normalizeFoodPoison(self, x_val):
        if x_val != 0:
            y_val = OBSERVATION_LINE_LENGTH - x_val
        else:
            y_val = x_val
        return y_val / (MAX_FOOD_DISTANCE - MIN_FOOD_DISTANCE) + MIN_FOOD_DISTANCE
                    
    """Updates entry 6 for eater"""
    def setEaterWallLine(self, dist):
        self.observationLines[5] = self._normalizeWall(dist)

    """Updates 1 to  5 entries for eater"""
    def setEaterFoodLines(self, food_vec):
        for i in range(len(food_vec)):
            self.observationLines[i] = self._normalizeFoodPoison(food_vec[i])
    
    """Updates entries 7 to 11 for eater"""
    def setEaterPoisonLines(self, poison_vec):
        for i in range(len(poison_vec)):
            self.observationLines[i+6] = self._normalizeFoodPoison(poison_vec[i])

    def getObservationLines(self):
        return self.observationLines
    
    def _outOfBounds(self, xPos, yPos):
        outOfBounds = False
        if xPos < SIDE_LIMIT or xPos > SCREEN_WIDTH - SIDE_LIMIT:
            outOfBounds = True
        elif yPos < SIDE_LIMIT or yPos > SCREEN_HEIGHT - SIDE_LIMIT:
            outOfBounds = True
        return outOfBounds

    def moveInBounds(self):
        if self._outOfBounds(self.xPos, self.yPos):  
            if self.xPos < SIDE_LIMIT:
                self.xPos = SIDE_LIMIT
            if self.yPos < SIDE_LIMIT:
                self.yPos = SIDE_LIMIT
            if self.xPos > SCREEN_WIDTH - SIDE_LIMIT:
                self.xPos = SCREEN_WIDTH - SIDE_LIMIT
            if self.yPos > SCREEN_HEIGHT - SIDE_LIMIT:
                self.yPos = SCREEN_HEIGHT - SIDE_LIMIT
 
"""Class handling the center-lines wall-distance detection"""     
class WallDetector(object):
    def __init__(self):
        self.WE = 10
    
    """Detects if walls are in proximity"""
    def nearbyWall(self, eater):
        nearby = False
        if eater.xPos < OBSERVATION_LINE_LENGTH or eater.xPos > SCREEN_WIDTH - OBSERVATION_LINE_LENGTH \
                or eater.yPos < OBSERVATION_LINE_LENGTH or eater.yPos > SCREEN_HEIGHT - OBSERVATION_LINE_LENGTH:
            nearby = True 
        return nearby

    def observeWall(self, eater):
        centerObsLine = [OBSERVATION_LINE_LENGTH*np.cos(degToRad(eater.direction)) + eater.xPos, \
                        OBSERVATION_LINE_LENGTH*np.sin(degToRad(eater.direction)) + eater.yPos]
        belowX = False
        belowY = False
        aboveX = False
        aboveY = False
        # Distance into wall
        dist = 0
        if centerObsLine[0] < self.WE:
            belowX = True
        elif centerObsLine[0] > SCREEN_WIDTH - self.WE:
            aboveX = True
        if centerObsLine[1] < self.WE:
            belowY = True
        elif centerObsLine[1] > SCREEN_HEIGHT - self.WE:
            aboveY = True
        # Calculate distance to the wall
        if belowX:
            if aboveY or belowY:
                dist = MAX_WALL_DISTANCE
            else:
                dist = -centerObsLine[0] + self.WE + EATER_OBSERVE_EXTRA
        elif aboveX:
            if aboveY or belowY:
                dist = MAX_WALL_DISTANCE        
            else:
                dist = centerObsLine[0] - SCREEN_WIDTH + self.WE + EATER_OBSERVE_EXTRA
        elif aboveY:
            dist = centerObsLine[1] - SCREEN_HEIGHT + self.WE + EATER_OBSERVE_EXTRA
        elif belowY:
            dist = abs(centerObsLine[1]) + self.WE + EATER_OBSERVE_EXTRA
        return dist
    
class NeuralNetwork(object):
    def __init__(self):
        self.W1 = START_SIGMA*np.random.randn(NUMBER_OF_NODES, NUMBER_OF_INPUTS)
        self.W2 = START_SIGMA*np.random.randn(NUMBER_OF_OUTPUTS, NUMBER_OF_NODES)
            
    """Takes inputs and propagates into an output which is returned"""
    def propagate(self, inputVector, allInfo=False):
        z1 = np.dot(self.W1, inputVector)
        a2 = sigmoid(z1)
        z2 = np.dot(self.W2, a2)
        a3 = sigmoid(z2)
        # Generates the move-vector
        y = [False, False, False]
        for k in range(NUMBER_OF_OUTPUTS):
            if a3[k] > 0.5:
                y[k] = True
        if allInfo:
            return inputVector, a2, a3, y
        else:
            return y
        
    def setW(self, wNew):
        endW1 = NUMBER_OF_INPUTS*NUMBER_OF_NODES
        self.W1 = wNew[0:endW1].reshape(NUMBER_OF_NODES, NUMBER_OF_INPUTS)
        self.W2 = wNew[endW1:].reshape(NUMBER_OF_OUTPUTS, NUMBER_OF_NODES)
        
class Evolver(object):
    def __init__(self, eaters):
        self.eaters = eaters
        self.generation = 1
        self.noOfMutations = 0
    
    """Finds the two eaters who got the highest scores, combines them and
    sets eater with lowest score as the new one"""
    def evolve(self, eaters):
        scores = [Eater.getScore() for Eater in self.eaters]
        indLow = scores.index(min(scores))
        ind1 = scores.index(max(scores))
        scores[ind1] = 0
        ind2 = scores.index(max(scores))
        eater1 = self.eaters[ind1]
        eater2 = self.eaters[ind2]
        wEater1 = np.concatenate((eater1.NeuralNetwork.W1.ravel(), eater1.NeuralNetwork.W2.ravel()))
        wEater2 = np.concatenate((eater2.NeuralNetwork.W1.ravel(), eater2.NeuralNetwork.W2.ravel()))
        # Artificial meosis creates a child
        split = np.random.uniform(low=0, high=1)
        wNew = split*wEater1 + (1-split)*wEater2
        eaters[indLow].NeuralNetwork.setW(wNew)
        if VISUALIZATIONS:
            eaters[indLow].setColor(False, True, False)
            eater1.setColor(True, False, False)
            eater2.setColor(True, False, False)
        if WITH_MUTATION:
            # Each eater has 10% Chance to mutate
            for Eater in eaters:
                r = np.random.uniform(low=0, high=1)
                if r < MUTATE_PROBABILITY:
                    self._addMuate()
                    wEater = np.concatenate((Eater.NeuralNetwork.W1.ravel(), Eater.NeuralNetwork.W2.ravel()))
                    wEater += 2*np.random.randn()*np.random.randn((len(wEater)))
                    Eater.NeuralNetwork.setW(wEater)
                    Eater.setColor(False, False, True)
        self.generation += 1
        
    def _addMuate(self):
        self.noOfMutations += 1
    
    def resetMutations(self):
        self.noOfMutations = 0

class EaterHandler(object):
    def __init__(self):
        if KEYBOARD == False:
            self.eaters = [Eater() for _ in range(NUMBER_OF_EATERS)]
        else:
            self.eaters = [Eater()]
        
    def getFirstEater(self):
        return self.eaters[0]
    
    def newPositions(self):
        for eater in self.eaters:
            eater.newPosition()
            
    def setDirections(self):
        for Eater in self.eaters:
            if WD.nearbyWall(Eater):
                Eater.setEaterWallLine(WD.observeWall(Eater))
            else:
                Eater.setEaterWallLine(0)
            Eater.setEaterFoodLines(FPH.findFoodDistances(Eater))
            Eater.setEaterPoisonLines(FPH.findPoisonDistances(Eater))
    
    def keyBoardMove(self):
        for Eater in self.eaters:
            Eater.eaterMove(keyboardInputs())
            
    def neuralMove(self):
        for Eater in self.eaters:
            Eater.eaterMove(Eater.NeuralNetwork.propagate(Eater.getObservationLines()))
    
    def moveAndDetect(self, FoodPoisonHandler):
        for Eater in self.eaters:
            Eater.moveInBounds()
            FoodPoisonHandler.detectFoodEaten(Eater)
            FoodPoisonHandler.detectPoisonFound(Eater)
    
    def getTotScore(self):
        totScore = 0
        for eater in self.eaters:
            totScore += eater.score
        return totScore
    
    def drawEaters(self, Drawer):
        for Eater in self.eaters:
            Drawer.drawEater(Eater)
    
    def resetEaters(self):
        for Eater in self.eaters:
            Eater.resetColor()
            Eater.newPosition()
            
    def resetScores(self):
        for Eater in self.eaters:
            Eater.resetScore()
    

def keyboardInputs():
    keys = pg.key.get_pressed()
    move = [False,False,False]   
    if keys[pg.K_UP]:
        move[2] = True
    if keys[pg.K_LEFT]:
        move[0] = True
    if keys[pg.K_RIGHT]:
        move[1] = True
    return move
       

# --Setup--
pg.init()
if SHOW_BRAIN:
    size = (SCREEN_WIDTH + BRAIN_WIDTH, SCREEN_HEIGHT)
else:
    size = (SCREEN_WIDTH, SCREEN_HEIGHT)
screen = pg.display.set_mode(size)
pg.display.set_caption('Eater evolution game')

quitButton = False
# When x-button is pressed or game has completed
done = False
# Becomes true after t_r seconds when new eater positions
reset = False
# When 2*t_r seconds have passed
generationComplete = False

generationTotScore = []
generationMutations = []

clock = pg.time.Clock()    

# All the classes used
EH = EaterHandler()
FPH = FoodPoisonHandler()
D = Drawer(screen)
WD = WallDetector()
Ev = Evolver(EH.eaters)

if SHOW_BRAIN:
    D.setBrainEater(EH.getFirstEater())
    D.setNewBrainColors()
    
"""Main game loop"""
for _ in range(NUMBER_OF_GENERATIONS):
    t0 = time.time()
    
    while not done:
        # Find out if reset or finished
        t1 = time.time()
        if t1 - t0 > RESET_TIME and reset == False:
            EH.newPositions()
            reset = True
        if t1 - t0 > GAME_TIME:
            done = True
            generationComplete = True
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
                quitButton = True
        # Set all detection vectors
        EH.setDirections()
        # Move using keyboard or neural networks
        if KEYBOARD == True:
            EH.keyBoardMove()
            
        if KEYBOARD == False:
            EH.neuralMove()
        
        EH.moveAndDetect(FPH)
    
        # Create new foods if eaten
        if FPH.newFoodsRequired():
            FPH.updateFoodsList()
        if FPH.newPoisonRequired():
            FPH.updatePoisonList()
        
        totScore = EH.getTotScore()    
        # Draw everything on the screen
        screen.fill(LIGHT_GRAY)
        if SHOW_BRAIN:
            D.drawBrain()
        D.drawFoods(FPH.getFoodPositions())
        D.drawPoison(FPH.getPoisonPositions())
        D.makeScoreCounter(totScore)
        D.makeGenerationCounter(Ev)
        
        EH.drawEaters(D)
        
        #Drawer.makeFoodCounter(Eater)
        pg.display.flip()
        clock.tick(30)
    
    # After each generation
    if generationComplete:
        EH.resetEaters()
        Ev.evolve(EH.eaters)
        FPH.resetList()
        generationTotScore.append(totScore)
        generationMutations.append(Ev.noOfMutations)
        Ev.resetMutations()
        EH.resetScores()
        if SHOW_BRAIN:
            D.setNewBrainColors()
        done, reset, gameComplete = False, False, False
    
    if quitButton:
        break
pg.quit()


x = range(1, len(generationTotScore)+1)

print("\n\n\t############  DIAGNOSTICS  #############\n\n")
plt.figure(1)
plt.bar(x, generationTotScore)
plt.xlabel('Generation')
plt.ylabel('Total foods collected')
plt.figure(2)
plt.bar(x, generationMutations)
plt.xlabel('Generation')
plt.ylabel('Number of mutations')

        
        
        
    