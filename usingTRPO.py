from sb3_contrib import TRPO

from env import SumoEnv

e = SumoEnv(gui=True, noWarnings=True, epLen=500, bunched=True, traffic=10)

model = TRPO.load("trafficEplen250/mixedConfigs/reward_paper_lowest10_NoExp/trpo_Nobusvectorinstate_NoWaitTime")
# model = TRPO.load("final/trpoNoTraffic")

obs = e.reset()
while True:
    action, states = model.predict(obs, deterministic=True)
    print("Action: ", action)
    obs, reward, done, info = e.step(action)
    if done:
      break

e.close()