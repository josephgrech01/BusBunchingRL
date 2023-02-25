from stable_baselines3 import PPO

from env import SumoEnv

e = SumoEnv(gui=True, noWarnings=True)

model = PPO.load("eplen250/ppo_Ring_Speed60NewHeadway_Nobusvectorinstate_NoWaitTime_Alpha1_PaperHeadwayExpNo2")

obs = e.reset()
# print("obs: ", obs)

while True:
# for _ in range(250):
    action, states = model.predict(obs)
    print("action: ", action)
    obs, reward, done, info = e.step(action)
    if done:
        # obs = e.reset()
        break


e.close()
