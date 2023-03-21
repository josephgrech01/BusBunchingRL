from stable_baselines3 import PPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

# e.reset() # added to continue training 

model = PPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/WithoutBusVector/TrafficEplen250/mixedConfigs/reward_paper_normalized5320_lowest10_NoExp/ppo")#learning_starts=2500, exploration_initial_eps=2)
# model = PPO.load("eplen250/reward_paper_normalized/ppo_Speed60_Nobusvectorinstate_NoWaitTime")

# model.set_env(e) # added to continue training

model.learn(total_timesteps=250000, log_interval=1)
model.save("trafficEplen250/mixedConfigs/reward_paper_normalized5320_lowest10_NoExp/ppo_Nobusvectorinstate_NoWaitTime")

e.close()
