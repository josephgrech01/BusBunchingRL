from env import SumoEnv
import random

env = SumoEnv(gui=True, noWarnings=True)
episodes = 3
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