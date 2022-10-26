import gym

from stable_baselines3 import DQN, dqn

from env import SumoEnv

e = SumoEnv()

model = DQN("MlpPolicy", e, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_model")

del model
model = DQN.load("dqn_model")

obs = e.reset()

while True:
    action, states = model.predict(obs, deterministic=True)
    obs, reward, done, info = e.step(action)
    if done:
        obs = e.reset()
