import gym

from stable_baselines3 import DQN, dqn

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

model = DQN("MlpPolicy", e, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_model")

e.close()

# del model
# model = DQN.load("dqn_model")

# obs = e.reset()



# while True:
# for _ in range(250):
#     action, states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = e.step(action)
#     if done:
#         obs = e.reset()

# e.close()
