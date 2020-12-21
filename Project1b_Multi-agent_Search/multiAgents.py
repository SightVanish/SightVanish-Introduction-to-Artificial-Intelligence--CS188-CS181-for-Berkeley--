# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        "*** YOUR CODE HERE ***"

        agent_number = gameState.getNumAgents() - 1 # index of agents, starting from 0 (paceman)

        # input the current state of paceman and the depth level it will consider
        def pacemanLogic(state, level):
            # paceman has reached the bottom of its evaluation, return evaluated value
            if state.isWin() or state.isLose() or level >= self.depth:
                return self.evaluationFunction(state) 

            # return the max value of all possible movements
            result = float("-inf")
            # get all possible movements
            for action in state.getLegalActions(0):
                # note that we only consider ghost 1 here, and behavior of ghost 1 is influenced by ghost2
                result = max(result, ghostLogic(state.generateSuccessor(0, action), level, 1))
            return result

        # input the current state of paceman and the depth level it will consider and ghost index (starting from 1)
        def ghostLogic(state, level, ghost_index):
            # ghost has reached the bottom of its evaluation, return evaluated value
            if state.isWin() or state.isLose() or level >= self.depth:
                return self.evaluationFunction(state)

            # return the min value of all possible movements
            result = float("inf")
            for action in state.getLegalActions(ghost_index):
                # if the ghost is the last one and next level is the bottom level, return the min of the evaluated value
                if ghost_index == agent_number:
                    if level == self.depth - 1:
                        result = min(result, self.evaluationFunction(state.generateSuccessor(ghost_index, action)))
                # if the ghost is the last one and next level is paceman, return the min of the paceman value
                    else:
                        result = min(result, pacemanLogic(state.generateSuccessor(ghost_index, action), level + 1)) # note that level++ here, level only increases here
                # if the ghost is not the last one, return the min of next ghost value
                else:
                    result = min(result, ghostLogic(state.generateSuccessor(ghost_index, action), level, ghost_index + 1))
            return result
        
        # now we are paceman, so we need to consider the max value of possible solution which is given by the min value of ghost
        # init solution

        # following code does not work
        '''
        solution = gameState.getLegalActions(0)[0]
        value = 0
        # go through all its possible action
        for action in gameState.getLegalActions(0):
            # choose the max, note that the init level is 0!
            v = ghostLogic(gameState.generateSuccessor(0, action), 0, 1)
            if v >= value:
                value = v
                solution = action
        return solution
        '''

        solution = {}
        for action in gameState.getLegalActions(0):
            solution[action] = ghostLogic(gameState.generateSuccessor(0, action), 0, 1) # note level = 0 not 1
        # sort dictionary
        solution = sorted(solution.items(),key = lambda x:x[1],reverse = True)
        return solution[0][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agent_number = gameState.getNumAgents() - 1 # index of agents, starting from 0 (paceman)

        def successor_value(state, level, agent_index, alpha, beta):
            if state.isWin() or state.isLose() or level >= self.depth:
                return self.evaluationFunction(state) 
            if agent_index == 0:
                return max_value(state, level, alpha, beta)
            else:
                return min_value(state, level, agent_index, alpha, beta)

            
        # return agent value
        def max_value(state, level, alpha, beta):
            v = float("-inf")
            # go through all action choices
            for action in state.getLegalActions(0):
                # get successor value
                v = max(v, successor_value(state.generateSuccessor(0, action), level, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        # return ghost value
        def min_value(state, level, ghost_index, alpha, beta):
            v = float("inf")
            # go through all action choices
            for action in state.getLegalActions(ghost_index):
                if ghost_index == agent_number:
                    # if it is the last ghost, then its successor, paceman, will increase its level
                    v = min(v, successor_value(state.generateSuccessor(ghost_index, action), level + 1, 0, alpha, beta))
                else:
                    # else, we go to next ghost
                    v = min(v, successor_value(state.generateSuccessor(ghost_index, action), level, ghost_index + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        # following part is similar to max_value
        # init alpha & bet
        alpha = float("-inf")
        beta = float("inf")
        # init result value to -inf -- cause we are considering the strategy of paceman
        v = float("-inf")
        # init next movement to stop
        final_action = 'Stop'

        # go through all possible action
        for action in gameState.getLegalActions(0):
            # note level = 0, agent is the first ghost
            value = successor_value(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            # update result and action
            if value > v:
                v = value
                final_action = action
            alpha = max(alpha, v)
        # note: return action instead of v
        return final_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # similar with MinimaxAgent, only modifying ghostLogic
        agent_number = gameState.getNumAgents() - 1 # index of agents, starting from 0 (paceman)
        # input the current state of paceman and the depth level it will consider
        def pacemanLogic(state, level):
            # paceman has reached the bottom of its evaluation, return evaluated value
            if state.isWin() or state.isLose() or level >= self.depth:
                return self.evaluationFunction(state) 

            # return the max value of all possible movements
            result = float("-inf")
            # get all possible movements
            for action in state.getLegalActions(0):
                # note that we only consider ghost 1 here, and behavior of ghost 1 is influenced by ghost2
                result = max(result, ghostLogic(state.generateSuccessor(0, action), level, 1))
            return result

        # input the current state of paceman and the depth level it will consider and ghost index (starting from 1)
        def ghostLogic(state, level, ghost_index):
            # ghost has reached the bottom of its evaluation, return evaluated value
            if state.isWin() or state.isLose() or level >= self.depth:
                return self.evaluationFunction(state)

            # return the min value of all possible movements
            
            result = 0.0
            count = 0
            for action in state.getLegalActions(ghost_index):
                # if the ghost is the last one and next level is the bottom level, return the min of the evaluated value
                count += 1
                if ghost_index == agent_number:
                    if level == self.depth - 1:
                        result += self.evaluationFunction(state.generateSuccessor(ghost_index, action))
                # if the ghost is the last one and next level is paceman, return the min of the paceman value
                    else:
                        result += pacemanLogic(state.generateSuccessor(ghost_index, action), level + 1) # note that level++ here, level only increases here
                # if the ghost is not the last one, return the min of next ghost value
                else:
                    result += ghostLogic(state.generateSuccessor(ghost_index, action), level, ghost_index + 1)
            return result / count
        # init
        v = float("-inf")
        final_action = "STOP"

        solution = {}
        # go through all possible actions
        for action in gameState.getLegalActions(0):
            # init level to 0
            solution[action] = ghostLogic(gameState.generateSuccessor(0, action), 0, 1)
        # sort dictionary
        solution = sorted(solution.items(),key = lambda x:x[1],reverse = True)
        return solution[0][0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION:
        Assign different elements with different weight. Then caculate the linear sum of all values. note I use eciprocal of distance to ghost / food here, as mentioned in Hint.

        And something weird happens: I've got 3/6 firstly. It showed Paceman crashed, but i did not know why. I waited a few minutes and did nothing, then it showed 0/6 then 6/6 ...
   
    """
    "*** YOUR CODE HERE ***"

    # init
    value = 0.0
    Pacman_pos = currentGameState.getPacmanPosition()
    # only 1 ghost
    ghost_state = currentGameState.getGhostStates()
    food_pos = currentGameState.getFood().asList()
    pellet_pos = currentGameState.getCapsules()

    # weights
    food_weight = 5
    pellet_weight = 10
    ghost_weight = -10
    ghost_scared_weight = 1000 # we can do whatever we want now
    # note: we cannot assign inf here
    
    # assign score--very important
    value += currentGameState.getScore()

    # assign ghost value
    for ghost in ghost_state:
        distance = manhattanDistance(Pacman_pos, ghost.getPosition())
        # set death 
        if distance <= 0:
            return float("-inf")
        if ghost.scaredTimer > 0.0:
            value += ghost_scared_weight / distance
        else:
            value += ghost_weight / distance

    # assign food value
    for food in food_pos:
        value += food_weight / manhattanDistance(Pacman_pos, food)

    # assign pellet value
    for pellet in pellet_pos:
        value += pellet_weight / manhattanDistance(pellet, food)

    return value



# Abbreviation
better = betterEvaluationFunction
