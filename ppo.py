from stable_baselines3 import PPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

model = PPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/WithoutBusVector/eplen250/reward/ppo")#learning_starts=2500, exploration_initial_eps=2)
model.learn(total_timesteps=180000, log_interval=1)
model.save("eplen250/reward_0_2/ppo_Speed60_Nobusvectorinstate_NoWaitTime")

e.close()
