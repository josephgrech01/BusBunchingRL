from env import SumoEnv

env = SumoEnv(gui=True, noWarnings=True, epLen=500, traffic=10, bunched=True)

episodes = 1
for episode in range(1, episodes + 1):  

    state = env.reset()

    done = False

    while not done:
        forwardHeadway, backwardHeadway = env.getHeadways()
        # choose action based on forward and backward headways
        if forwardHeadway <= 443.33:
            action = 0
        elif backwardHeadway <= 443.33:
            action = 1
        else:
            action = 2
        state, reward, done, info = env.step(action)

env.close()
