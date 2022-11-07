from env import SumoEnv

env = SumoEnv(gui=True)
episodes = 3
for episode in range(1, episodes + 1):

    state = env.reset()

    done = False
    score = 0

    while not done:
        state, reward, done, info = env.step(2)
        score += reward

    print("Episode: {} Score: {}".format(episode, score))

env.close()