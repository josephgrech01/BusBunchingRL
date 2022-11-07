import gym

from stable_baselines3 import DQN, dqn

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

model = DQN("MlpPolicy", e, verbose=1, exploration_initial_eps=2)
model.learn(total_timesteps=30000, log_interval=4)
model.save("dqn_model")

e.close()
