from sb3_contrib import TRPO

from env import SumoEnv

e = SumoEnv(gui=True, noWarnings=True, bunched=False, traffic=True)

model = TRPO.load("trafficEplen250/mixedConfigs/reward_paper_lowest10_NoExp/trpo_Nobusvectorinstate_NoWaitTime")

obs = e.reset()
while True:
    action, states = model.predict(obs, deterministic=True)
    print("Action: ", action)
    obs, reward, done, info = e.step(action)
    if done:
      break

e.close()