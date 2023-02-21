from stable_baselines3 import PPO

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

model = PPO("MlpPolicy", e, verbose=1, learning_rate=0.001, tensorboard_log="tensorboard/WithoutBusVector/eplen250/ppoAlpha1")#learning_starts=2500, exploration_initial_eps=2)
model.learn(total_timesteps=180000, log_interval=1)
model.save("eplen250/ppo_Ring_Speed60NewHeadway_Nobusvectorinstate_NoWaitTime_Alpha1")
# model.save("dqn_model_test_old")

e.close()
