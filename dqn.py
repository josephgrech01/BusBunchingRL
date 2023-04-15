from stable_baselines3 import DQN

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True, epLen=250, traffic=0, mixedConfigs=False, bunched=False)

model = DQN("MlpPolicy", e, verbose=1, exploration_initial_eps=1, learning_rate=0.001, tensorboard_log="tensorboard/final/dqnNoTraffic")#learning_starts=2500, exploration_initial_eps=2)
model.learn(total_timesteps=180000, log_interval=1)
model.save("final/dqnNoTraffic")
# model.save("dqn_model_test_old")

e.close()
