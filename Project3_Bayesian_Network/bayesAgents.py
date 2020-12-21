# bayesAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import bayesNet as bn
import game
from game import Actions, Agent, Directions
import inference
import layout
import factorOperations
import itertools
import operator as op
import random
import util
import inference

from hunters import GHOST_COLLISION_REWARD, WON_GAME_REWARD
from layout import PROB_BOTH_TOP, PROB_BOTH_BOTTOM, PROB_ONLY_LEFT_TOP, \
    PROB_ONLY_LEFT_BOTTOM, PROB_FOOD_RED, PROB_GHOST_RED

X_POS_VAR = "xPos"
FOOD_LEFT_VAL = "foodLeft"
GHOST_LEFT_VAL = "ghostLeft"
X_POS_VALS = [FOOD_LEFT_VAL, GHOST_LEFT_VAL]

Y_POS_VAR = "yPos"
BOTH_TOP_VAL = "bothTop"
BOTH_BOTTOM_VAL = "bothBottom"
LEFT_TOP_VAL = "leftTop"
LEFT_BOTTOM_VAL = "leftBottom"
Y_POS_VALS = [BOTH_TOP_VAL, BOTH_BOTTOM_VAL, LEFT_TOP_VAL, LEFT_BOTTOM_VAL]

FOOD_HOUSE_VAR = "foodHouse"
GHOST_HOUSE_VAR = "ghostHouse"
HOUSE_VARS = [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR]

TOP_LEFT_VAL = "topLeft"
TOP_RIGHT_VAL = "topRight"
BOTTOM_LEFT_VAL = "bottomLeft"
BOTTOM_RIGHT_VAL = "bottomRight"
HOUSE_VALS = [TOP_LEFT_VAL, TOP_RIGHT_VAL, BOTTOM_LEFT_VAL, BOTTOM_RIGHT_VAL]

OBS_VAR_TEMPLATE = "obs(%d,%d)"

BLUE_OBS_VAL = "blue"
RED_OBS_VAL = "red"
NO_OBS_VAL = "none"
OBS_VALS = [BLUE_OBS_VAL, RED_OBS_VAL, NO_OBS_VAL]

ENTER_LEFT = 0
ENTER_RIGHT = 1
EXPLORE = 2

def constructBayesNet(gameState):
    """
    You *must* name all position and house variables using the constants
    (X_POS_VAR, FOOD_HOUSE_VAR, etc.) at the top of this file. 
    """

    obsVars = []
    edges = []
    variableDomainsDict = {}

    "*** YOUR CODE HERE ***"
    # - a single "x position" variable (controlling the x pos of the houses)
    variableDomainsDict[X_POS_VAR] = X_POS_VALS # set x pos domain
    # - a single "y position" variable (controlling the y pos of the houses)
    variableDomainsDict[Y_POS_VAR] = Y_POS_VALS # set y pos domain
    # - a single "food house" variable (containing the house centers)
    variableDomainsDict[FOOD_HOUSE_VAR] = HOUSE_VALS # set food pos domain
    # - a single "ghost house" variable (containing the house centers)
    variableDomainsDict[GHOST_HOUSE_VAR] = HOUSE_VALS # set ghost pos domain

    # - a large number of "observation" variables for each cell Pacman can measure
    for housePos in gameState.getPossibleHouses():
        for obsPos in gameState.getHouseWalls(housePos):
            # get full set of observation variables
            obsVar = OBS_VAR_TEMPLATE % obsPos
            obsVars.append(obsVar)
    # - populate `edges` with every edge in the Bayes Net (a tuple `(from, to)`)
    edges.append((X_POS_VAR, GHOST_HOUSE_VAR))
    edges.append((X_POS_VAR, FOOD_HOUSE_VAR))
    edges.append((Y_POS_VAR, FOOD_HOUSE_VAR))
    edges.append((Y_POS_VAR, GHOST_HOUSE_VAR))
    for item in obsVars:
        variableDomainsDict[item] = OBS_VALS # set domain of obsVar
        edges.append((FOOD_HOUSE_VAR, item))
        edges.append((GHOST_HOUSE_VAR, item))

    # return
    variables = [X_POS_VAR, Y_POS_VAR] + HOUSE_VARS + obsVars
    net = bn.constructEmptyBayesNet(variables, edges, variableDomainsDict)

    return net, obsVars

def fillCPTs(bayesNet, gameState):
    fillXCPT(bayesNet, gameState)
    fillYCPT(bayesNet, gameState)
    fillHouseCPT(bayesNet, gameState)
    fillObsCPT(bayesNet, gameState)

def fillXCPT(bayesNet, gameState):
    from layout import PROB_FOOD_LEFT 
    xFactor = bn.Factor([X_POS_VAR], [], bayesNet.variableDomainsDict())
    xFactor.setProbability({X_POS_VAR: FOOD_LEFT_VAL}, PROB_FOOD_LEFT)
    xFactor.setProbability({X_POS_VAR: GHOST_LEFT_VAL}, 1 - PROB_FOOD_LEFT)
    bayesNet.setCPT(X_POS_VAR, xFactor)

def fillYCPT(bayesNet, gameState):
    """
    Question 2a: Bayes net probabilities

    Fill the CPT that gives the prior probability over the y position variable.
    See the definition of `fillXCPT` above for an example of how to do this.
    You can use the PROB_* constants imported from layout rather than writing
    probabilities down by hand.
    """
    yFactor = bn.Factor([Y_POS_VAR], [], bayesNet.variableDomainsDict())
    "*** YOUR CODE HERE ***"
    yFactor.setProbability({Y_POS_VAR: BOTH_TOP_VAL}, PROB_BOTH_TOP)
    yFactor.setProbability({Y_POS_VAR: BOTH_BOTTOM_VAL}, PROB_BOTH_BOTTOM)
    yFactor.setProbability({Y_POS_VAR: LEFT_TOP_VAL}, PROB_ONLY_LEFT_TOP)
    yFactor.setProbability({Y_POS_VAR: LEFT_BOTTOM_VAL}, PROB_ONLY_LEFT_BOTTOM)
    bayesNet.setCPT(Y_POS_VAR, yFactor)

def fillHouseCPT(bayesNet, gameState):
    foodHouseFactor = bn.Factor([FOOD_HOUSE_VAR], [X_POS_VAR, Y_POS_VAR], bayesNet.variableDomainsDict())
    for assignment in foodHouseFactor.getAllPossibleAssignmentDicts():
        left = assignment[X_POS_VAR] == FOOD_LEFT_VAL
        top = assignment[Y_POS_VAR] == BOTH_TOP_VAL or \
                (left and assignment[Y_POS_VAR] == LEFT_TOP_VAL)

        if top and left and assignment[FOOD_HOUSE_VAR] == TOP_LEFT_VAL or \
                top and not left and assignment[FOOD_HOUSE_VAR] == TOP_RIGHT_VAL or \
                not top and left and assignment[FOOD_HOUSE_VAR] == BOTTOM_LEFT_VAL or \
                not top and not left and assignment[FOOD_HOUSE_VAR] == BOTTOM_RIGHT_VAL:
            prob = 1
        else:
            prob = 0

        foodHouseFactor.setProbability(assignment, prob)
    bayesNet.setCPT(FOOD_HOUSE_VAR, foodHouseFactor)

    ghostHouseFactor = bn.Factor([GHOST_HOUSE_VAR], [X_POS_VAR, Y_POS_VAR], bayesNet.variableDomainsDict())
    for assignment in ghostHouseFactor.getAllPossibleAssignmentDicts():
        left = assignment[X_POS_VAR] == GHOST_LEFT_VAL
        top = assignment[Y_POS_VAR] == BOTH_TOP_VAL or \
                (left and assignment[Y_POS_VAR] == LEFT_TOP_VAL)

        if top and left and assignment[GHOST_HOUSE_VAR] == TOP_LEFT_VAL or \
                top and not left and assignment[GHOST_HOUSE_VAR] == TOP_RIGHT_VAL or \
                not top and left and assignment[GHOST_HOUSE_VAR] == BOTTOM_LEFT_VAL or \
                not top and not left and assignment[GHOST_HOUSE_VAR] == BOTTOM_RIGHT_VAL:
            prob = 1
        else:
            prob = 0

        ghostHouseFactor.setProbability(assignment, prob)
    bayesNet.setCPT(GHOST_HOUSE_VAR, ghostHouseFactor)

# translate string to certain position
def getPosition(pos, gameState):
    bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = gameState.getPossibleHouses()
    if pos == 'topLeft':
        return topLeftPos
    elif pos == 'topRight':
        return topRightPos
    elif pos == 'bottomLeft':
        return bottomLeftPos
    elif pos == 'bottomRight':
        return bottomRightPos

def fillObsCPT(bayesNet, gameState):
    """
    Question 2b: Bayes net probabilities

    Fill the CPT that gives the probability of an observation in each square,
    given the locations of the food and ghost houses. Refer to the project
    description for what this probability table looks like. You can use
    PROB_FOOD_RED and PROB_GHOST_RED from the top of the file.

    You will need to create a new factor for *each* of 4*7 = 28 observation
    variables. Don't forget to call bayesNet.setCPT for each factor you create.

    The XXXPos variables at the beginning of this method contain the (x, y)
    coordinates of each possible house location.

    IMPORTANT:
    Because of the particular choice of probabilities higher up in the Bayes
    net, it will never be the case that the ghost house and the food house are
    in the same place. However, the CPT for observations must still include a
    vaild probability distribution for this case. To conform with the
    autograder, use the *food house distribution* over colors when both the food
    house and ghost house are assigned to the same cell.
    """
    bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = gameState.getPossibleHouses()
    housesPos = gameState.getPossibleHouses()
    "*** YOUR CODE HERE ***"
    # wall position for all different possible wall pos
    wallPos = list(gameState.getHouseWalls(bottomLeftPos))\
            + list(gameState.getHouseWalls(topLeftPos))\
            + list(gameState.getHouseWalls(bottomRightPos))\
            + list(gameState.getHouseWalls(topRightPos))
    for wall in wallPos:
        # get observation variable
        obsVar = OBS_VAR_TEMPLATE % wall
        # create observation variable factor--factor is a table
        obsVarFactor = bn.Factor([obsVar], [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR], bayesNet.variableDomainsDict())
        # find the nearest possible wall
        nearestWall = housesPos[0]
        for j in housesPos:
            if abs(wall[0]-j[0]) + abs(wall[1]-j[1]) < abs(wall[0]-nearestWall[0]) + abs(wall[1]-nearestWall[1]):
                nearestWall = j

        for item in obsVarFactor.getAllPossibleAssignmentDicts():
            color = item[obsVar]
            p = 0
            # if nearest is food
            if nearestWall == getPosition(item['foodHouse'], gameState):
                if color == RED_OBS_VAL:
                    p = PROB_FOOD_RED
                elif color == BLUE_OBS_VAL:
                    p = 1 - PROB_FOOD_RED
            # if nearest is ghost
            elif nearestWall == getPosition(item['ghostHouse'], gameState):
                if color == RED_OBS_VAL:
                    p = PROB_GHOST_RED
                elif color == BLUE_OBS_VAL:
                    p = 1 - PROB_GHOST_RED
            # if neither food / ghost
            else:
                if color == NO_OBS_VAL:
                    p = 1
            # set probability
            obsVarFactor.setProbability(item, p)
        bayesNet.setCPT(obsVar, obsVarFactor)  


def getMostLikelyFoodHousePosition(evidence, bayesNet, eliminationOrder):
    """
    Question 7: Marginal inference for pacman

    Find the most probable position for the food house.
    First, call the variable elimination method you just implemented to obtain
    p(FoodHouse | everything else). Then, inspect the resulting probability
    distribution to find the most probable location of the food house. Return
    this.

    (This should be a very short method.)
    """
    "*** YOUR CODE HERE ***"
    p = 0
    assign = 0
    query = inference.inferenceByVariableElimination(bayesNet, FOOD_HOUSE_VAR, evidence, eliminationOrder)
    for i in query.getAllPossibleAssignmentDicts():
        current_p = query.getProbability(i)
        if current_p > p:
            p = current_p
            assign = i
    return assign

class BayesAgent(game.Agent):

    def registerInitialState(self, gameState):
        self.bayesNet, self.obsVars = constructBayesNet(gameState)
        fillCPTs(self.bayesNet, gameState)

        self.distances = cacheDistances(gameState)
        self.visited = set()
        self.steps = 0

    def getAction(self, gameState):
        self.visited.add(gameState.getPacmanPosition())
        self.steps += 1

        if self.steps < 40:
            return self.getRandomAction(gameState)
        else:
            return self.goToBest(gameState)

    def getRandomAction(self, gameState):
        legal = list(gameState.getLegalActions())
        legal.remove(Directions.STOP)
        random.shuffle(legal)
        successors = [gameState.generatePacmanSuccessor(a).getPacmanPosition() for a in legal]
        ls = [(a, s) for a, s in zip(legal, successors) if s not in gameState.getPossibleHouses()]
        ls.sort(key=lambda p: p[1] in self.visited)
        return ls[0][0]

    def getEvidence(self, gameState):
        evidence = {}
        for ePos, eColor in gameState.getEvidence().items():
            obsVar = OBS_VAR_TEMPLATE % ePos
            obsVal = {
                "B": BLUE_OBS_VAL,
                "R": RED_OBS_VAL,
                " ": NO_OBS_VAL
            }[eColor]
            evidence[obsVar] = obsVal
        return evidence

    def goToBest(self, gameState):
        evidence = self.getEvidence(gameState)
        unknownVars = [o for o in self.obsVars if o not in evidence]
        eliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR, GHOST_HOUSE_VAR]
        bestFoodAssignment = getMostLikelyFoodHousePosition(evidence, 
                self.bayesNet, eliminationOrder)

        tx, ty = dict(
            zip([BOTTOM_LEFT_VAL, TOP_LEFT_VAL, BOTTOM_RIGHT_VAL, TOP_RIGHT_VAL],
                gameState.getPossibleHouses()))[bestFoodAssignment[FOOD_HOUSE_VAR]]
        bestAction = None
        bestDist = float("inf")
        for action in gameState.getLegalActions():
            succ = gameState.generatePacmanSuccessor(action)
            nextPos = succ.getPacmanPosition()
            dist = self.distances[nextPos, (tx, ty)]
            if dist < bestDist:
                bestDist = dist
                bestAction = action
        return bestAction

class VPIAgent(BayesAgent):

    def __init__(self):
        BayesAgent.__init__(self)
        self.behavior = None
        NORTH = Directions.NORTH
        SOUTH = Directions.SOUTH
        EAST = Directions.EAST
        WEST = Directions.WEST
        self.exploreActionsRemaining = \
                list(reversed([NORTH, NORTH, NORTH, NORTH, EAST, EAST, EAST,
                    EAST, SOUTH, SOUTH, SOUTH, SOUTH, WEST, WEST, WEST, WEST]))

    def reveal(self, gameState):
        bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = \
                gameState.getPossibleHouses()
        for housePos in [bottomLeftPos, topLeftPos, bottomRightPos]:
            for ox, oy in gameState.getHouseWalls(housePos):
                gameState.data.observedPositions[ox][oy] = True

    def getExplorationProbsAndOutcomes(self, evidence):
        unknownVars = [o for o in self.obsVars if o not in evidence]
        assert len(unknownVars) == 7
        assert len(set(evidence.keys()) & set(unknownVars)) == 0
        firstUnk = unknownVars[0]
        restUnk = unknownVars[1:]

        unknownVars = [o for o in self.obsVars if o not in evidence]
        eliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR]
        houseMarginals = inference.inferenceByVariableElimination(self.bayesNet,
                [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR], evidence, eliminationOrder)

        probs = [0 for i in range(8)]
        outcomes = []
        for nRed in range(8):
            outcomeVals = [RED_OBS_VAL] * nRed + [BLUE_OBS_VAL] * (7 - nRed)
            outcomeEvidence = dict(zip(unknownVars, outcomeVals))
            outcomeEvidence.update(evidence)
            outcomes.append(outcomeEvidence)

        for foodHouseVal, ghostHouseVal in [(TOP_LEFT_VAL, TOP_RIGHT_VAL),
                (TOP_RIGHT_VAL, TOP_LEFT_VAL)]:

            condEvidence = dict(evidence)
            condEvidence.update({FOOD_HOUSE_VAR: foodHouseVal, 
                GHOST_HOUSE_VAR: ghostHouseVal})
            assignmentProb = houseMarginals.getProbability(condEvidence)

            oneObsMarginal = inference.inferenceByVariableElimination(self.bayesNet,
                    [firstUnk], condEvidence, restUnk + [X_POS_VAR, Y_POS_VAR])

            assignment = oneObsMarginal.getAllPossibleAssignmentDicts()[0]
            assignment[firstUnk] = RED_OBS_VAL
            redProb = oneObsMarginal.getProbability(assignment)

            for nRed in range(8):
                outcomeProb = combinations(7, nRed) * \
                        redProb ** nRed * (1 - redProb) ** (7 - nRed)
                outcomeProb *= assignmentProb
                probs[nRed] += outcomeProb

        return list(zip(probs, outcomes))


    def getAction(self, gameState):

        if self.behavior == None:
            self.reveal(gameState)
            evidence = self.getEvidence(gameState)
            unknownVars = [o for o in self.obsVars if o not in evidence]
            enterEliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR]
            exploreEliminationOrder = [X_POS_VAR, Y_POS_VAR]

            print (evidence)
            print (enterEliminationOrder)
            print (exploreEliminationOrder)
            enterLeftValue, enterRightValue = \
                    self.computeEnterValues(evidence, enterEliminationOrder)
            exploreValue = self.computeExploreValue(evidence,
                    exploreEliminationOrder)

            # TODO double-check
            enterLeftValue -= 4
            enterRightValue -= 4
            exploreValue -= 20

            bestValue = max(enterLeftValue, enterRightValue, exploreValue)
            if bestValue == enterLeftValue:
                self.behavior = ENTER_LEFT
            elif bestValue == enterRightValue:
                self.behavior = ENTER_RIGHT
            else:
                self.behavior = EXPLORE

            # pause 1 turn to reveal the visible parts of the map
            return Directions.STOP

        if self.behavior == ENTER_LEFT:
            return self.enterAction(gameState, left=True)
        elif self.behavior == ENTER_RIGHT:
            return self.enterAction(gameState, left=False)
        else:
            return self.exploreAction(gameState)

    def enterAction(self, gameState, left=True):
        bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = \
                gameState.getPossibleHouses()

        dest = topLeftPos if left else topRightPos

        actions = gameState.getLegalActions()
        neighbors = [gameState.generatePacmanSuccessor(a) for a in actions]
        neighborStates = [s.getPacmanPosition() for s in neighbors]
        best = min(zip(actions, neighborStates), 
                key=lambda x: self.distances[x[1], dest])
        return best[0]

    def exploreAction(self, gameState):
        if self.exploreActionsRemaining:
            return self.exploreActionsRemaining.pop()

        evidence = self.getEvidence(gameState)
        enterLeftValue, enterRightValue = self.computeEnterValues(evidence,
                [X_POS_VAR, Y_POS_VAR])

        if enterLeftValue > enterRightValue:
            self.behavior = ENTER_LEFT
            return self.enterAction(gameState, left=True)
        else:
            self.behavior = ENTER_RIGHT
            return self.enterAction(gameState, left=False)

def cacheDistances(state):
    width, height = state.data.layout.width, state.data.layout.height
    states = [(x, y) for x in range(width) for y in range(height)]
    walls = state.getWalls().asList() + state.data.layout.redWalls.asList() + state.data.layout.blueWalls.asList()
    states = [s for s in states if s not in walls]
    distances = {}
    for i in states:
        for j in states:
            if i == j:
                distances[i, j] = 0
            elif util.manhattanDistance(i, j) == 1:
                distances[i, j] = 1
            else:
                distances[i, j] = 999999
    for k in states:
        for i in states:
            for j in states:
                if distances[i,j] > distances[i,k] + distances[k,j]:
                    distances[i,j] = distances[i,k] + distances[k,j]

    return distances

# http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def combinations(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer / denom
