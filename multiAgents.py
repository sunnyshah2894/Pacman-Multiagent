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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        """
        Get the list of current food positions
        """
        currFoodPositions = currentGameState.getFood().asList()

        """
        Part 1: Find the minimum distance to the closest food point. I have used the manhattan distance to calculate 
                the minimum distance. 
                This is one part of our score
        """

        infinity = 99999999999
        score = infinity

        for foodPoint in currFoodPositions:
            score = min(score, manhattanDistance(foodPoint, newPos))

        """
        Part 2: Even if we can eat the food, we need to check if we are safe from ghost. Following are the conditions 
                on which we measure the score:
                1)  If the manhattanDistance to ghost is 1, and ghost is not scared, then we should penalize heavily
                2)  If the manhattanDistance to ghost is 2, and ghost is not scared, then we should penalize heavily
                3)  (Manhattan distance + scaredTimer for the ghost) is zero, we penalize heavily, since step 4 below will 
                    fail
                4)  For all the other options, we can see that the chance of getting killed by ghost is inversely 
                    proportional to (scareTimer of the ghost and manhattanDistance of the ghost)  
        """
        for ghost in newGhostStates:
            manhattanDistanceFromGhost = manhattanDistance(ghost.getPosition(), newPos)
            if manhattanDistanceFromGhost == 1 and ghost.scaredTimer <= 0:
                score += infinity
            if manhattanDistanceFromGhost == 2 and ghost.scaredTimer <= 0:
                score += infinity
            if ( ghost.scaredTimer + manhattanDistance(ghost.getPosition(), newPos)) <= 0:
                score += infinity
            else:
                score += 1/( ghost.scaredTimer + manhattanDistance(ghost.getPosition(), newPos))

        """
            Now, our score metrics above is mixture of minDistance to closest food and relative chance of getting killed 
            from ghost. Thus, score will be low for a good action. But we need to output a big value for good actions.
            
            We do this by negating the score and then returning the negated value
        """
        return ( 0 - score )

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
      Your minimax agent (question 2)
    """
    totalNodesExpandedTillNow = 0
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        # Find the number of agents. This is (1 + number of ghosts)
        numberOfAgents = gameState.getNumAgents()

        # Starting from depth 0
        currDepth = 0

        # Agent property that will be updated every time we find a valid optimal action
        self.nextActionToTake = Directions.STOP


        # Start from MAX turn
        self.max_value(gameState, currDepth, self.depth, numberOfAgents)

        # print "totalNodesExpandedTillNow = ", self.totalNodesExpandedTillNow

        # Return the property which contains the best action till now.
        return self.nextActionToTake

    def value(self, state, currDepth, maxDepthToReach, agent, numberOfAgents ):
        """

            There can be 2 possible actions:
                1)  If we have went through all the ghosts, then we will receive agent value as numberOfAgents.
                    This means that, we have finished expanding the plies for MIN and we should now call max_value.
                    In doing so, we should also increase the depth, as completing the MIN for all ghosts, means one
                    total move.
                2)  Agent is between 1 and numberOfAgents-1, indicating we are still in the same move, where we
                    are evaluating the actions for each ghost.

        """
        self.totalNodesExpandedTillNow += 1
        if agent % numberOfAgents == 0:
            return self.max_value( state, currDepth+1, maxDepthToReach, numberOfAgents )
        else:
            return self.min_value( state,currDepth, maxDepthToReach, agent, numberOfAgents )

    def max_value(self, state, currDepth, maxDepthToReach, numberOfAgents):

        """
        :param state: current state of the game
        :param currDepth: current depth of the ply we explored
        :param numberOfAgents: Number of agents in the game
        :param maxDepthToReach: Maximum depth we should explore. Any node at this depth should be directly evaluated using the self.evaluation function
        :return: max_score that the MAX player can achieve.

        -------------------------------------------------------

        This method tries to find the max score a MAX player can make.
        It is based on the following logic:
            1)  If the currDepth is the maximum depth according to the self.depth, then directly calculate the score for
                the state using the self.evaluationFunction and the return the score.

            2)  Calculate the max score based on the scores of the MIN players for every action taken by the MAX player.

            3)  If we cannot find any optimal max score (this may be probably because of state either being a win or a lose position),
                we directly send the score of that state using the state.getScore() method

            4)  In case, we found the max score, than we update the self.nextActionToTake property, which will be used
                by our getAction method.

        """

        # If the currDepth is the maximum depth according to the self.depth, then directly calculate the score for
        # the state using the self.evaluationFunction and the return the score.
        if (currDepth ) == maxDepthToReach:
            return self.evaluationFunction(state)


        listOfActions = state.getLegalActions(0)

        # Start with a very low value for the max_score, which we will try to maximize
        max_score = -99999999999

        # Stores the best action to take from the current state
        best_action_so_far = Directions.STOP

        # boolean flag to keep note of whether we found any eligible action
        best_action_found = False

        # Calculate the max score based on the scores of the MIN players for every action taken by the MAX player.
        for action in listOfActions:

            # Call the min_value(value function will decide ) with agent = 1, which is the first ghost.
            mini_score = self.value(state.generateSuccessor(0, action), currDepth, self.depth, 1, numberOfAgents)

            # If the min_score is less than the value found till now. Update out max_score and note the best action
            if mini_score > max_score:
                max_score = mini_score
                best_action_so_far = action
                best_action_found = True

        # If we cannot find any optimal max score (this may be probably because of state either being a win or a lose position),
        # we directly send the score of that state using the state.getScore() method
        if not best_action_found:
            return state.getScore()

        # In case, we found the max score, than we update the self.nextActionToTake property, which will be used
        # by our getAction method.
        self.nextActionToTake = best_action_so_far

        # Return the max_score
        return max_score

    def min_value(self, state, currDepth, maxDepthToReach, agent, numberOfAgents):

        """

        :param state: current state of the game
        :param currDepth: current depth of the ply we explored
        :param maxDepthToReach: maximum depth we should reach to
        :param agent: Agent Id of the ghost.
        :param numberOfAgents: Number of agents in the game
        :return: max_score that the MAX player (ghost agent) can achieve.

        This method tries to find the min score a MIN player (ghost agent) can make.
        It is based on the following logic:
            1)  Calculate the min score recursively for each action. A action taken by agent should consult the next agent for its
                score and similarly till we traverse all the MIN agents.

            2)  If we cannot find any optimal min score (this may be probably because of state either being a win or a lose position),
                we directly send the score of that state using the state.getScore() method

            3)  In case, we found the min score, we return the value. This will be the minimum score the ghosts(adversaries)
                will try to make to compete against pacman.

        """

        legalActionsFromGhost = state.getLegalActions(agent)

        # start with a large min value
        min_score = 99999999999

        # Calculate the min score recursively for each action. A action taken by agent should consult the next agent for its
        # score and similarly till we traverse all the MIN agents.
        for action in legalActionsFromGhost:

            # Find the successor state using the state.generateSuccessor method
            successor = state.generateSuccessor(agent, action)

            # If there are no agents remaining, we call the max_value to calculate the scores for the next depth
            # Else we call the min_value for the next ghost.
            # Our value function will take care of this
            min_score = min(min_score, self.value(successor, currDepth, maxDepthToReach, agent + 1, numberOfAgents))

        # If we couldn't find any min_score, we return the state.getScore() instead (this may be probably because of state either being a win or a lose position)
        if min_score == 99999999999:
            return state.getScore()

        # return the min_score achieved by the agent
        return min_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    totalNodesExpandedTillNow = 0
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Find the number of agents. This is (1 + number of ghosts)
        numberOfAgents=gameState.getNumAgents()

        # Starting from depth 0
        currDepth=0

        # Agent property that will be updated every time we find a valid optimal action
        self.nextActionToTake=Directions.STOP
        alpha = -99999999999
        beta = 99999999999

        # Start from MAX turn
        self.max_value(gameState, currDepth, self.depth, numberOfAgents , alpha, beta)

        # print "totalNodesExpandedTillNow = ", self.totalNodesExpandedTillNow

        # Return the property which contains the best action till now.
        return self.nextActionToTake

    def value(self, state, currDepth, maxDepthToReach, agent, numberOfAgents, alpha, beta ):
        """

            There can be 2 possible actions:
                1)  If we have went through all the ghosts, then we will receive agent value as numberOfAgents.
                    This means that, we have finished expanding the plies for MIN and we should now call max_value.
                    In doing so, we should also increase the depth, as completing the MIN for all ghosts, means one
                    total move.
                2)  Agent is between 1 and numberOfAgents-1, indicating we are still in the same move, where we
                    are evaluating the actions for each ghost.

        """
        self.totalNodesExpandedTillNow += 1
        if agent % numberOfAgents == 0:
            return self.max_value( state, currDepth+1, maxDepthToReach, numberOfAgents, alpha, beta )
        else:
            return self.min_value( state,currDepth, maxDepthToReach, agent, numberOfAgents, alpha, beta )

    def max_value(self, state, currDepth, maxDepthToReach, numberOfAgents, alpha, beta):

        """
        :param state: current state of the game
        :param currDepth: current depth of the ply we explored
        :param numberOfAgents: Number of agents in the game
        :param maxDepthToReach: Maximum depth we should explore. Any node at this depth should be directly evaluated using the self.evaluation function
        :return: max_score that the MAX player can achieve.

        -------------------------------------------------------

        This method tries to find the max score a MAX player can make.
        It is based on the following logic:
            1)  If the currDepth is the maximum depth according to the self.depth, then directly calculate the score for
                the state using the self.evaluationFunction and the return the score.

            2)  Calculate the max score based on the scores of the MIN players for every action taken by the MAX player.

                2.1) Check if the max score so far has crossed the beta value. If yes, then return, since the MIN player will
                     not be using this value.

            3)  If we cannot find any optimal max score (this may be probably because of state either being a win or a lose position),
                we directly send the score of that state using the state.getScore() method

            4)  In case, we found the max score, than we update the self.nextActionToTake property, which will be used
                by our getAction method.

        """

        # If the currDepth is the maximum depth according to the self.depth, then directly calculate the score for
        # the state using the self.evaluationFunction and the return the score.
        if (currDepth) == maxDepthToReach:
            return self.evaluationFunction(state)

        listOfActions=state.getLegalActions(0)

        # Start with a very low value for the max_score, which we will try to maximize
        max_score=-99999999999

        # Stores the best action to take from the current state
        best_action_so_far=Directions.STOP

        # boolean flag to keep note of whether we found any eligible action
        best_action_found=False

        # Calculate the max score based on the scores of the MIN players for every action taken by the MAX player.
        for action in listOfActions:

            # Call the value(which will in turn call min_value) with agent = 1, which is the first ghost.
            mini_score=self.value(state.generateSuccessor(0, action), currDepth, self.depth, 1, numberOfAgents, alpha, beta)

            # If the min_score is less than the value found till now. Update out max_score and note the best action
            if mini_score > max_score:
                max_score=mini_score
                best_action_so_far=action
                best_action_found=True

            # if the max score so far is greater than the current beta value, then the MIN player is never
            # going to use this value. So stop expanding further
            if max_score > beta:
                break

            # Update the alpha value
            alpha = max(alpha,max_score)

        # If we cannot find any optimal max score (this may be probably because of state either being a win or a lose position),
        # we directly send the score of that state using the state.getScore() method
        if not best_action_found:
            return state.getScore()

        # In case, we found the max score, than we update the self.nextActionToTake property, which will be used
        # by our getAction method.
        self.nextActionToTake=best_action_so_far

        # Return the max_score
        return max_score

    def min_value(self, state, currDepth, maxDepthToReach, agent, numberOfAgents, alpha, beta):

        """

        :param state: current state of the game
        :param currDepth: current depth of the ply we explored
        :param maxDepthToReach: maximum depth we should reach to
        :param agent: Agent Id of the ghost.
        :param numberOfAgents: Number of agents in the game
        :return: max_score that the MAX player (ghost agent) can achieve.

        This method tries to find the min score a MIN player (ghost agent) can make.
        It is based on the following logic:
            1)  Calculate the min score recursively for each action. A action taken by agent should consult the next agent for its
                score and similarly till we traverse all the MIN agents.

            2)  If we cannot find any optimal min score (this may be probably because of state either being a win or a lose position),
                we directly send the score of that state using the state.getScore() method

                2.1) Check if the min score so far has gone below the alpha value. If yes, then return, since the MAX player will
                     not be using this value.

            3)  In case, we found the min score, we return the value. This will be the minimum score the ghosts(adversaries)
                will try to make to compete against pacman.

        """

        legalActionsFromGhost=state.getLegalActions(agent)

        # start with a large min value
        min_score=99999999999

        # Calculate the min score recursively for each action. A action taken by agent should consult the next agent for its
        # score and similarly till we traverse all the MIN agents.
        for action in legalActionsFromGhost:

            # Find the successor state using the state.generateSuccessor method
            successor=state.generateSuccessor(agent, action)

            # If there are no agents remaining, we call the max_value to calculate the scores for the next depth
            # Else we call the min_value for the next ghost.
            # Our value function will take care of this
            min_score=min(min_score, self.value(successor, currDepth, maxDepthToReach, agent + 1, numberOfAgents, alpha, beta))

            # IF min_score is below alpha, then the MAX have already found a path with max score,
            # so do not expand further
            if min_score < alpha:
                break

            # update beta till now
            beta = min( beta, min_score )

        # If we couldn't find any min_score, we return the state.getScore() instead (this may be probably because of state either being a win or a lose position)
        if min_score == 99999999999:
            return state.getScore()

        # return the min_score achieved by the agent
        return min_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    totalNodesExpandedTillNow = 0
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # Find the number of agents. This is (1 + number of ghosts)
        numberOfAgents=gameState.getNumAgents()

        # Starting from depth 0
        currDepth=0

        # Agent property that will be updated every time we find a valid optimal action
        self.nextActionToTake=Directions.STOP
        alpha = -99999999999
        beta = 99999999999

        # Start from MAX turn
        self.max_value(gameState, currDepth, self.depth, numberOfAgents)

        # print "totalNodesExpandedTillNow = ", self.totalNodesExpandedTillNow

        # Return the property which contains the best action till now.
        return self.nextActionToTake

    def value(self, state, currDepth, maxDepthToReach, agent, numberOfAgents):
        """

            There can be 2 possible actions:
                1)  If we have went through all the ghosts, then we will receive agent value as numberOfAgents.
                    This means that, we have finished expanding the plies for CHANCE nodes and we should now call max_value.
                    In doing so, we should also increase the depth, as completing the CHANCE for all ghosts, means one
                    total move.
                2)  Agent is between 1 and numberOfAgents-1, indicating we are still in the same move, where we
                    are evaluating the actions for each ghost, which will be a chance_value for ghost.

        """
        self.totalNodesExpandedTillNow += 1
        if agent % numberOfAgents == 0:
            return self.max_value( state, currDepth+1, maxDepthToReach, numberOfAgents)
        else:
            return self.chance_value( state,currDepth, maxDepthToReach, agent, numberOfAgents)

    def max_value(self, state, currDepth, maxDepthToReach, numberOfAgents):

        """
        :param state: current state of the game
        :param currDepth: current depth of the ply we explored
        :param numberOfAgents: Number of agents in the game
        :param maxDepthToReach: Maximum depth we should explore. Any node at this depth should be directly evaluated using the self.evaluation function
        :return: max_score that the MAX player can achieve.

        -------------------------------------------------------

        This method tries to find the max score a MAX player can make.
        It is based on the following logic:
            1)  If the currDepth is the maximum depth according to the self.depth, then directly calculate the score for
                the state using the self.evaluationFunction and the return the score.

            2)  Calculate the max score based on the scores of the MIN players for every action taken by the MAX player.

            3)  If we cannot find any optimal max score (this may be probably because of state either being a win or a lose position),
                we directly send the score of that state using the state.getScore() method

            4)  In case, we found the max score, than we update the self.nextActionToTake property, which will be used
                by our getAction method.

        """

        # If the currDepth is the maximum depth according to the self.depth, then directly calculate the score for
        # the state using the self.evaluationFunction and the return the score.
        if (currDepth) == maxDepthToReach:
            return self.evaluationFunction(state)

        listOfActions=state.getLegalActions(0)

        # Start with a very low value for the max_score, which we will try to maximize
        max_score=-99999999999

        # Stores the best action to take from the current state
        best_action_so_far=Directions.STOP

        # boolean flag to keep note of whether we found any eligible action
        best_action_found=False

        # Calculate the max score based on the scores of the MIN players for every action taken by the MAX player.
        for action in listOfActions:

            # Call the min_value with agent = 1, which is the first ghost.
            mini_score=self.value(state.generateSuccessor(0, action), currDepth, self.depth, 1, numberOfAgents)

            # If the min_score is less than the value found till now. Update out max_score and note the best action
            if mini_score > max_score:
                max_score=mini_score
                best_action_so_far=action
                best_action_found=True


        # If we cannot find any optimal max score (this may be probably because of state either being a win or a lose position),
        # we directly send the score of that state using the state.getScore() method
        if not best_action_found:
            return state.getScore()

        # In case, we found the max score, than we update the self.nextActionToTake property, which will be used
        # by our getAction method.
        self.nextActionToTake=best_action_so_far

        # Return the max_score
        return max_score

    def chance_value(self, state, currDepth, maxDepthToReach, agent, numberOfAgents):

        """

        :param state: current state of the game
        :param currDepth: current depth of the ply we explored
        :param maxDepthToReach: maximum depth we should reach to
        :param agent: Agent Id of the ghost.
        :param numberOfAgents: Number of agents in the game
        :return: max_score that the MAX player (ghost agent) can achieve.

        This method tries to find the min score a MIN player (ghost agent) can make.
        It is based on the following logic:
            1)  Calculate the min score recursively for each action. A action taken by agent should consult the next agent for its
                score and similarly till we traverse all the MIN agents.

            2)  Store the score from each action in a list.

            3)  If we cannot find any optimal min score (this may be probably because of state either being a win or a lose position),
                we directly send the score of that state using the state.getScore() method

            4)  Now since the ghosts play randomly, we can find the expected score
                as the sum of scores divided by the number of actions.

        """

        legalActionsFromGhost=state.getLegalActions(agent)

        # start with a large min value
        scores = []

        # Calculate the min score recursively for each action. A action taken by agent should consult the next agent for its
        # score and similarly till we traverse all the MIN agents.
        for action in legalActionsFromGhost:

            # Find the successor state using the state.generateSuccessor method
            successor=state.generateSuccessor(agent, action)

            # If there are no agents remaining, we call the max_value to calculate the scores for the next depth
            # Else we call the chance_value for the next ghost.
            # Our value function will take care of this
            scores.append(self.value(successor, currDepth, maxDepthToReach, agent + 1, numberOfAgents))


        # If we couldn't find any min_score, we return the state.getScore() instead (this may be probably because of state either being a win or a lose position)
        if len(scores) == 0:
            return state.getScore()

        # return the expected score as sum of all scores divided by the number of legal actions.
        return float(sum(scores))/len(scores)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

