from sb3_contrib import TRPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True, epLen=250, traffic=10, mixedConfigs=True, bunched=False)

model = TRPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/trpoTraffic")

model.learn(total_timesteps=250000, log_interval=1)
model.save("models/trpoTraffic")

e.close()