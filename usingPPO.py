from stable_baselines3 import PPO

from env import SumoEnv

e = SumoEnv(gui=True, noWarnings=True, epLen=500, traffic=10, bunched=False)

# model = PPO.load("eplen250/ppo_Ring_Speed60NewHeadway_Nobusvectorinstate_NoWaitTime_Alpha1_PaperHeadway.zip") #no traffic
# model = PPO.load("eplen250/reward_0_2/ppo_Speed60_Nobusvectorinstate_NoWaitTimeCONTINUED")
# model = PPO.load("trafficEplen250/reward_paper_lowest10_NoExp/ppo_Nobusvectorinstate_NoWaitTime") #old model (not bunched)
# model = PPO.load("trafficEplen250/bunched/reward_paper_lowest10_NoExp/ppo_Nobusvectorinstate_NoWaitTime") #new model (bunched)
# model = PPO.load("trafficEplen250/bunched/reward_paper_normalized5320_lowest10_NoExp/ppo_Nobusvectorinstate_NoWaitTime") #new model normalized (bunched)
model = PPO.load("trafficEplen250/mixedConfigs/reward_paper_lowest10_NoExp/ppo_Nobusvectorinstate_NoWaitTime.zip") #new model (mixed configs)
# model = PPO.load("trafficEplen250/mixedConfigs/reward_paper_normalized5320_lowest10_NoExp/ppo_Nobusvectorinstate_NoWaitTime") #new model normalized (mixed configs)

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
