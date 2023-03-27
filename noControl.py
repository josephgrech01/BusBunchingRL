from env import SumoEnv
import random

env = SumoEnv(gui=True, noWarnings=True, traffic=False, bunched=False)

episodes = 1
for episode in range(1, episodes + 1):

    state = env.reset()

    done = False
    score = 0

    while not done:
        action = random.randint(0,2)
        state, reward, done, info = env.step(2)#action)
        score += reward

    print("Episode: {} Score: {}".format(episode, score))

env.close()

# env.reset()   #the observation returned by reset method must be a numpy array
# from gym.utils.env_checker import check_env
# check_env(env)
# print("hey")