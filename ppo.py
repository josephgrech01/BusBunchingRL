from stable_baselines3 import PPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True, epLen=250, traffic=10, mixedConfigs=True, bunched=False)

model = PPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/ppoTraffic")

model.learn(total_timesteps=250000, log_interval=1)
model.save("models/ppoTraffic")

e.close()