from stable_baselines3 import DQN

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

model = DQN("MlpPolicy", e, verbose=1, exploration_initial_eps=1, learning_rate=0.001, tensorboard_log="tensorboard/")#learning_starts=2500, exploration_initial_eps=2)
model.learn(total_timesteps=250000, log_interval=4)
model.save("dqn_model_Ring_Speed60_eplen_750_NewHeadway")
# model.save("dqn_model_test_old")

e.close()
