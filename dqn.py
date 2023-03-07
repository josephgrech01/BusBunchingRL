from stable_baselines3 import DQN

from env import SumoEnv

e = SumoEnv(gui=False, noWarnings=True)

model = DQN("MlpPolicy", e, verbose=1, exploration_initial_eps=1, learning_rate=0.001, tensorboard_log="tensorboard/WithoutBusVector/eplen250/reward_paper_normalized5320NoExp/dqn")#learning_starts=2500, exploration_initial_eps=2)
model.learn(total_timesteps=180000, log_interval=4)
model.save("eplen250/reward_paper_normalized5320_NoExp/dqn_Speed60_Nobusvectorinstate_NoWaitTime")
# model.save("dqn_model_test_old")

e.close()
