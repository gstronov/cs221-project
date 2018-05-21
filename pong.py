import math, random
from collections import defaultdict
import gym
from gym import wrappers, logger
import numpy

import collections, random


#Qlearning class taken from  Blackjack assignment
class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, explorationProb):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state, done):
    	if done:
    		return 0
        self.numIters += 1
        if random.random() < self.explorationProb:
        	for i in range(35, 195):
	            row = state[1][i]
	            #ball is in row
	            if [236, 236, 236] in row:
	                return 2 #go up
	            #paddle is in row
	            if [92, 186, 92] in row:
	                return 3 #go down
	        #do nothing
	        return 0
        else:
            return max((self.getQ(state, action), action) for action in self.actions)[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        if newState != None:
            Vopt = max(self.getQ(newState, a) for a in self.actions)
            for feature, value in self.featureExtractor(state, action):
                self.weights[feature] -= self.getStepSize()*(self.getQ(state, action) - (reward + self.discount*Vopt))*value
        # END_YOUR_CODE

    def getWeights(self)
    	return self.weights

def getxy(obs, pixel):
	for i in range(35, 195):
		row = obs[i]
		for j in range(len(row)):
			if row[j].tolist() == pixel:
				ballxy = [i, j]
				return ballxy
	return [0, 0]

def round(x):
	divided = (int(x)/10)*10
	if math.fabs(x - divided) > math.fabs(x -(divided + 10)):
		return divided + 10
	return divided

#state is tuple of prevObservation, curObservation
def featureExtractorXY(state, action):
	features = []
	prevBall = getxy(state[0], [236, 236, 236])
	curBall = getxy(state[1], [236, 236, 236])
	agentPaddle = getxy(state[1], [92, 186, 92])[0]
	opPaddle = getxy(state[1], [213, 130,74])[0]
	xvelocity = curBall[1] - prevBall[1]
	yvelocity = curBall[0] - prevBall[0]

	if curBall[0] - agentPaddle > 0:
		features.append((("ballBelow", action), 1))
	else:
		features.append((("ballAbove", action), 1))

	if prevBall != [0, 0]:
		features.append((("prevBall", round(prevBall[0]), round(prevBall[1]), action), 1))
	if curBall != [0, 0]:
		features.append((("curBall", round(curBall[0]), round(curBall[1]), action), 1))
	features.append((("agentPaddle", round(agentPaddle), action), 1))
	features.append((("opPaddle", round(opPaddle), action), 1))
	features.append((("vx", round(xvelocity), action), 1))
	features.append((("vy", round(yvelocity), action), 1))
	return features


class QLearningAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done, q):
    	return q.getAction(observation, done)



if __name__ == '__main__':
    logger.set_level(logger.WARN)

    #Opens a Pong environment
    env = gym.make('Pong-v0')

    #directory to output game statistics
    outdir = 'tmp/results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QLearningAgent(env.action_space)

    num_games = 1000
    reward = 0
    done = False
    q = QLearningAlgorithm([0, 2, 3], discount = 1, featureExtractor = featureExtractorXY, explorationProb=0.3)

    #plays num_games number of games 
    for i in range(num_games):
        new_observation = env.reset()
        observation = new_observation
        prev_obs = observation
        while True:
            #env.render() #allows video of game in progress to be shown 
            action = agent.act((observation, new_observation), reward, done, q)
            prev_obs = observation
            observation = new_observation
            new_observation, reward, done, _ = env.step(action) #go to next action
            q.incorporateFeedback((prev_obs, observation), action, reward, (observation, new_observation))
            #the end of the game has been reached
            if done:
                break
    	print q.getWeights()
    #Closes the environment            
    env.close()