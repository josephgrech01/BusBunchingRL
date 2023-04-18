from env import SumoEnv
import random

env = SumoEnv(gui=True, noWarnings=True, epLen=500, traffic=10, bunched=True)

episodes = 1
for episode in range(1, episodes + 1):  

    state = env.reset()

    done = False
    score = 0

    while not done:
        forwardHeadway, backwardHeadway = env.getHeadways()
        if forwardHeadway <= 443.33:
            action = 0
        elif backwardHeadway <= 443.33:
            action = 1
        else:
            action = 2
        state, reward, done, info = env.step(action)
        score += reward



env.close()

# env.reset()   #the observation returned by reset method must be a numpy array
# from gym.utils.env_checker import check_env
# check_env(env)
# print("hey")