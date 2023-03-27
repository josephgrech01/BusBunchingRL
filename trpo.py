from sb3_contrib import TRPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True, mixedConfigs=True)

model = TRPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/WithoutBusVector/TrafficEplen250/mixedConfigs/reward_paper_lowest10_NoExp/trpo")
model.learn(total_timesteps=250000, log_interval=1)
model.save("trafficEplen250/mixedConfigs/reward_paper_lowest10_NoExp/trpo_Nobusvectorinstate_NoWaitTime")

e.close()