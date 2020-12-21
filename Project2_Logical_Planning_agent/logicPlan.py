# logicPlan.py
# ------------
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


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
successor_nodes = []
class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')

    A_or_B = A | B
    not_A_iff_not_B_or_C = ~A % (~B | C)
    not_A_or_not_B_or_C = logic.disjoin(~A, ~B, C)
    sentence = logic.conjoin(A_or_B, not_A_iff_not_B_or_C, not_A_or_not_B_or_C)
    return sentence

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')

    s1 = C % (B | D)
    s2 = A >> (~B & ~D)
    s3 = ~(B & ~C) >> A
    s4 = ~D >> C
    sentence = logic.conjoin(s1, s2, s3, s4)
    return sentence

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    WumpusAlive_1 = logic.PropSymbolExpr("WumpusAlive[1]")
    WumpusAlive_0 = logic.PropSymbolExpr("WumpusAlive[0]")
    WumpusBorn_0 = logic.PropSymbolExpr("WumpusBorn[0]")
    WumpusKilled_0 = logic.PropSymbolExpr("WumpusKilled[0]")

    s1 = WumpusAlive_1 % ((WumpusAlive_0 & ~WumpusKilled_0) | (~WumpusAlive_0 & WumpusBorn_0))
    s2 = ~(WumpusAlive_0 & WumpusBorn_0)
    s3 = WumpusBorn_0
    sentence = logic.conjoin(s1, s2, s3)
    return sentence

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    cnf = logic.to_cnf(sentence)
    solution = logic.pycoSAT(cnf)
    # note: if no possible solution, return false
    return solution

def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    # note: 
    # >>> disjoin([(A&B),(B|C),(B&C)])
    # ((A & B) | (B | C) | (B & C))
    return logic.disjoin(literals)

def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    not_literals = []
    for i in literals:
        not_literals.append(~i)
    for i in range(len(not_literals)):
        for j in range(len(not_literals)):
            if i != j:
                if i == 0 and j == 1:
                    solution = not_literals[i] | not_literals[j]
                else:
                    solution = logic.conjoin(solution, not_literals[i] | not_literals[j])
    return solution

def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    not_literals = []
    for i in literals:
        not_literals.append(~i)

    for i in range(len(not_literals)):
        for j in range(len(not_literals)):
            if i != j:
                if i == 0 and j == 1:
                    solution1 = not_literals[i] | not_literals[j]
                else:
                    solution1 = logic.conjoin(solution1, not_literals[i] | not_literals[j])

    solution2 = logic.disjoin(literals)
    # based on most 1, except all false
    return solution1 & solution2

def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    # model is a dictionary
    # plan is a list
    plan = [] # init plan
    
    for key in model.keys():
        if model[key]:
            sentence = logic.PropSymbolExpr.parseExpr(key)
            if sentence[0] in actions:
                plan.append(sentence)
    l = len(plan)
    # sort list
    solution = []
    for i in range(l):
        item = plan[0]
        index = 0
        for j in range(len(plan)):
            if int(plan[j][1]) < int(item[1]):
                #print("yes")
                item = plan[j]
                index = j
        plan.pop(index)
        solution.append(item[0])
    return solution

def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    # init current/final position
    current = logic.PropSymbolExpr(pacman_str, x, y, t)
    test = []
    if walls_grid[x+1][y] != True:
        action = logic.PropSymbolExpr(pacman_str, x+1, y, t-1) & logic.PropSymbolExpr('West', t-1)
        test.append(action)
        successor_nodes.append((x+1, y, t))
    if walls_grid[x-1][y] != True:
        action2 = logic.PropSymbolExpr(pacman_str, x-1, y, t-1) & logic.PropSymbolExpr('East', t-1)
        test.append(action2)
        successor_nodes.append((x+1, y, t))
    if walls_grid[x][y+1] != True:
        action3 = logic.PropSymbolExpr(pacman_str, x, y+1, t-1) & logic.PropSymbolExpr('South', t-1)
        test.append(action3)
        successor_nodes.append((x+1, y, t))
    if walls_grid[x][y-1] != True:
        action4 = logic.PropSymbolExpr(pacman_str, x, y-1, t-1) & logic.PropSymbolExpr('North', t-1)
        test.append(action4)
        successor_nodes.append((x+1, y, t))
    return current % logic.disjoin(test)

def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    available_action = ['North', 'East', 'South', 'West'] # evacuate STOP
    "*** YOUR CODE HERE ***"
    # get start/goal loaction
    start = problem.getStartState()
    goal = problem.getGoalState()

    start_state = logic.PropSymbolExpr("P", start[0], start[1], 0)
    start_logic = []
    for i in range(1, width+1):
        for j in range(1, height+1):
            start_logic.append(logic.PropSymbolExpr("P", i, j, 0))
    start_logic = exactlyOne(start_logic)
    start_logic = logic.conjoin(start_logic, start_state)
    # note that we must assert any other position is not allowed, otherwise it will cause weird error

    t = 1 # initial time
    # note that time start from 1 but not 0
    path = False

    while not path:
        # goal state
        goal_state = logic.PropSymbolExpr("P", goal[0], goal[1], t+1)
        # build succession logic
        successor = []
        for k in range(1, t+2): # note here must be t+2
            for i in range(1, width + 1): # skip the most-outside wall
                for j in range(1, height + 1):
                    if not walls[i][j]: # if (x, y) is not a wall
                        successor.append(pacmanSuccessorStateAxioms(i, j, k, walls))
        successor = logic.conjoin(successor)

        action_taken = []
        for i in range(0, t + 1):# last one is t
            all_action = []
            for action in available_action:
                all_action += [logic.PropSymbolExpr(action, i)]
            # one step must take exactly one action
            action_taken.append(exactlyOne(all_action))
        # each step must take action
        each_step_action = logic.conjoin(action_taken)
        # assemble model
        model = logic.conjoin(start_logic, goal_state, each_step_action, successor)
        path = findModel(model) 
        # complicated init state & goal state & way to achieve goal & all possible action can be taken & all possible successors to any point 
        t += 1
    return extractActionSequence(path, available_action)

def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    available_action = ['North', 'East', 'South', 'West'] # evacuate STOP
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()[0]
    food = problem.getStartState()[1]

    start_state = logic.PropSymbolExpr("P", start[0], start[1], 0)
    start_logic = []
    for i in range(1, width+1):
        for j in range(1, height+1):
            start_logic.append(logic.PropSymbolExpr("P", i, j, 0))
    start_logic = exactlyOne(start_logic)
    start_logic = logic.conjoin(start_logic, start_state)
    # note that we must assert any other position is not allowed, otherwise it will cause weird error


    t = 1 # initial time
    # note that time start from 1 but not 0
    path = False

    while not path:
        # goal state -- go through all food point
        goal_state = []
        for i in range(width+2): # note that here must be +2
            for j in range(height+2): # go through all points
                if food[i][j]:
                    tmp = []
                    for k in range(t+1):
                        tmp.append(logic.PropSymbolExpr("P", i, j, k))
                    tmp = atLeastOne(tmp)
                    goal_state.append(tmp)
        goal_state = logic.conjoin(goal_state)

        # build succession logic
        successor = []
        for k in range(1, t+2): # note here must be t+2
            for i in range(1, width + 1): # skip the most-outside wall
                for j in range(1, height + 1):
                    if not walls[i][j]: # if (x, y) is not a wall
                        successor.append(pacmanSuccessorStateAxioms(i, j, k, walls))
        successor = logic.conjoin(successor)

        action_taken = []
        for i in range(0, t + 1):# last one is t
            all_action = []
            for action in available_action:
                all_action += [logic.PropSymbolExpr(action, i)]
            # one step must take exactly one action
            action_taken.append(exactlyOne(all_action))
        # each step must take action
        each_step_action = logic.conjoin(action_taken)


        # assemble model
        model = logic.conjoin(start_logic, goal_state, each_step_action, successor)
        path = findModel(model) 
        # complicated init state & goal state & way to achieve goal & all possible action can be taken & all possible successors to any point 
        t = t + 1

    return extractActionSequence(path, available_action)


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    