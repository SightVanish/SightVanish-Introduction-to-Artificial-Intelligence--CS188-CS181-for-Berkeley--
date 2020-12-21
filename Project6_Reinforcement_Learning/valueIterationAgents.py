# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for k in range(self.iterations):
            temp = util.Counter()
            for state in self.mdp.getStates():
                # print(state)
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    # print(state, actions)
                    value = -float("inf")
                    for action in actions:
                        value = max(value, self.getQValue(state, action))
                        temp[state] = value
                        # print(state, action, self.getQValue(state, action))
            self.values = temp
                # print(self.values)









    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for next_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            # Q = sum(R+decay*V)
            q += p * (reward + self.discount * self.getValue(next_state))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # note: remember to raise exit
        if self.mdp.isTerminal(state):
            return None
        # note: it must be set as -inf but not 0
        max_q = -float("inf")
        max_action = None
        for action in self.mdp.getPossibleActions(state):
            if self.getQValue(state, action) > max_q:
                max_action = action
                max_q = self.getQValue(state, action)
        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # iter for certain times
        n = 0
        for i in range(self.iterations):
            # choose state
            state = self.mdp.getStates()[n]
            # update n for next state
            n = n + 1
            if n == len(self.mdp.getStates()):
                n = 0
            # remember to handle exception
            if self.mdp.isTerminal(state):
                continue
            
            # set to -inf
            value = -float("inf")
            for action in self.mdp.getPossibleActions(state):
                if self.getQValue(state, action) > value:
                    value = self.getQValue(state, action)
            self.values[state] = value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Initialize an empty priority queue.
        priority_q = util.PriorityQueue()

        # For each non-terminal state s
        for state in self.mdp.getStates():
            # get max_q
            q = -float('inf')
            for action in self.mdp.getPossibleActions(state):
                if self.getQValue(state, action) > q:
                    q = self.getQValue(state, action)
            if self.mdp.isTerminal(state):
                continue
            # add state to priority_q
            priority_q.push(state, -abs(q - self.getValue(state)))

        # iter
        for i in range(self.iterations):
            if priority_q.isEmpty():
                break
            state = priority_q.pop()
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                value = -float("inf")
                for action in actions:
                    value = max(value, self.getQValue(state, action))
                self.values[state] = value

                # go over its pre_state
                for pre_state in self.mdp.getStates():
                    # instead of computing all successors, we test whether we can go from pre_state to state
                    a = 0
                    for action in self.mdp.getPossibleActions(pre_state):
                        for next_state in self.mdp.getTransitionStatesAndProbs(pre_state, action):
                            if next_state[0] == state:
                                a += 1
                    if a == 0:
                        continue
                    qValue = max([self.getQValue(pre_state, action) for action in self.mdp.getPossibleActions(pre_state)])
                    diff = abs(qValue - self.getValue(pre_state))
                    if diff > self.theta:
                        priority_q.update(pre_state, -1*diff)
