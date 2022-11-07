from stable_baselines3 import DQN

from env import SumoEnv

e = SumoEnv(gui=True)

model = DQN.load("dqn_model")

obs = e.reset()
# print("obs: ", obs)

while True:
# for _ in range(250):
    action, states = model.predict(obs, deterministic=True)
    print("action: ", action)
    obs, reward, done, info = e.step(action)
    if done:
        # obs = e.reset()
        break


e.close()
