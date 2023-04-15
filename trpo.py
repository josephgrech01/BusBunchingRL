from sb3_contrib import TRPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True, epLen=250, traffic=0, mixedConfigs=False, bunched=False)

model = TRPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/final/trpoNoTraffic")
model.learn(total_timesteps=180000, log_interval=1)
model.save("final/trpoNoTraffic")

e.close()