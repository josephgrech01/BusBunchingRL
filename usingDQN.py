from stable_baselines3 import DQN

from env import SumoEnv

e = SumoEnv(gui=True, noWarnings=True, epLen=500, traffic=10, bunched=True, mixedConfigs=False)

# model = DQN.load("dqn_model_Ring_Speed60_eplen_750_NewHeadway")
# model = DQN.load("eplen250/reward_paper_normalized5320_NoExp/dqn_Speed60_Nobusvectorinstate_NoWaitTime.zip")
model = DQN.load("final/dqnTraffic.zip")

obs = e.reset()

while True:
# for _ in range(250):
    action, states = model.predict(obs, deterministic=True)
    print("action: ", action)
    obs, reward, done, info = e.step(action)
    if done:
        # obs = e.reset()
        break


e.close()
