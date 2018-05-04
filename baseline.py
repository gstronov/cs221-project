import gym
from gym import wrappers, logger

'''The baseline agent picks an action after being given observation, reward, and done. 
The observation is an RGB image of the game, which is an array of shape (210, 160, 3).
Reward is a value of the reward. Done is a boolean if the game has finished. 
To choose an action, the agent starts at the topmost row of pixels that are part of the game,
row 35 of the RGB array, and checks if a pixel in the row has the values of the ball, if so 
the agent chooses to go up. If not, it then checks if a pixel in the row has the values of the paddle, if so 
the agent chooses to go down.'''
class BaselineAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        #game is over
        if done:
            return 0
        #loop through all rows of pixels that define the game board where the paddle can go
        for i in range(35, 195):
            row = observation[i]
            #ball is in row
            if [236, 236, 236] in row:
                return 2 #go up
            #paddle is in row
            if [92, 186, 92] in row:
                return 3 #go down
        #do nothing
        return 0

"""
Code below this point is adapted from the random_agent code given on OpenAI Gym. 
"""
if __name__ == '__main__':
    logger.set_level(logger.WARN)

    #Opens a Pong environment
    env = gym.make('Pong-v0')

    #directory to output game statistics
    outdir = 'tmp/results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = BaselineAgent(env.action_space)

    num_games = 100
    reward = 0
    done = False

    #plays num_games number of games 
    for i in range(num_games):
        observation = env.reset()
        while True:
            env.render() #allows video of game in progress to be shown 
            action = agent.act(observation, reward, done)
            observation, reward, done, _ = env.step(action) #go to next action
            #the end of the game has been reached
            if done:
                break

    #Closes the environment            
    env.close()
