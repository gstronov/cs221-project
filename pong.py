import math, random
from collections import defaultdict
import gym
from gym import wrappers, logger
import numpy as np
from numba import jit

import collections, random


#Qlearning class taken from  Blackjack assignment
class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, explorationProb):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = state[2]
        features = self.featureExtractor(state, action)
        for f, v in features:
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state, done):
        if done:
            return 0
        if random.random() < self.explorationProb/4:
            return random.choice([0, 2, 3])
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
        return .1

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

    def getWeights(self):
        return self.weights

    def setExplorationProbability(self, p):
        self.explorationProb = p

    def getExProb(self):
        return self.explorationProb

@jit(nopython=True, cache=True)
def getPositions(obs):
    answer = np.zeros(3)
    for i in range(195, 35, -1):
        for j in range(0, 160):
            #ball
            if obs[i, j, 0] == 236 and obs[i, j, 1] == 236 and obs[i, j, 2] == 236:
                answer[0] = i
                answer[1] = j
            #paddle
            elif obs[i, j, 0] == 92 and obs[i, j, 1] == 186 and obs[i, j, 2] ==  92:
                answer[2] = i
    return answer

#state is tuple of prevObservation, curObservation, score
def featureExtractorXY(state, action):
    features = []
    curPos = getPositions(state[1])
    prevPos = getPositions(state[0])
    
    xvelocity = curPos[1] - prevPos[1]
    yvelocity = curPos[0] - prevPos[0]

    if xvelocity > 0:
        features.append((("vx+", action), 1))
    else:
        features.append((("vx-", action), 1))
    
    if yvelocity > 0:
        features.append((("vy+", action), 1))
    else:
        features.append((("vy-", action), 1))
    
    features.append((("agentPaddle", action, curPos[2]), 1))
    features.append((("ballx", action, curPos[1]), 1))
    features.append((("bally", action, curPos[0]), 1))
    
    if curPos[0] - curPos[2] > 36:
        features.append((("ballBelow", action), 1))
    elif curPos[0] - curPos[2] > 0:
        features.append((("ballSame", action), 1))
    else:
        features.append((("ballAbove", action), 1))
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

    num_games = 10000
    reward = 0
    done = False
      
    q = QLearningAlgorithm([0, 2, 3], discount = 1, featureExtractor = featureExtractorXY, explorationProb=.8)

    #plays num_games number of games 
    for i in range(num_games):
        new_observation = env.reset()
        observation = new_observation
        prev_obs = observation
        score = 0
        while True:
            #env.render() #allows video of game in progress to be shown 
            action = agent.act((observation, new_observation, score), reward, done, q)
            prev_obs = observation
            observation = new_observation
            new_observation, reward, done, _ = env.step(action) #go to next action
            if reward == 1.0:
                score += 1
            q.incorporateFeedback((prev_obs, observation, score), action, reward, (observation, new_observation, score))
            #the end of the game has been reached
            if done:
                break
        if i % 100 == 0:
            print q.getWeights()
        if i % 500 == 0 and i != 0:
            curProb = q.getExProb()
            q.setExplorationProbability(max([curProb - .2, 0.1]))
        print(str(i) + " games completed with current game's score " + str(score))
    #Closes the environment            
    env.close()
