from stable_baselines3 import PPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

# e.reset() # added to continue training 

model = PPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/WithoutBusVector/eplen250/reward_paper_normalized5320NoExp/ppo")#learning_starts=2500, exploration_initial_eps=2)
# model = PPO.load("eplen250/reward_paper_normalized/ppo_Speed60_Nobusvectorinstate_NoWaitTime")

# model.set_env(e) # added to continue training

model.learn(total_timesteps=180000, log_interval=1)
model.save("eplen250/reward_paper_normalized5320_NoExp/ppo_Speed60_Nobusvectorinstate_NoWaitTime")

e.close()
