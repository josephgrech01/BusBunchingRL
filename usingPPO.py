from stable_baselines3 import PPO

from env import SumoEnv

actions = ['Hold', 'Skip', 'Proceed']
e = SumoEnv(gui=True, noWarnings=True, epLen=500, traffic=10, bunched=True)

# no traffic
# model = PPO.load("models/ppoNoTraffic")

# traffic
model=PPO.load("models/ppoTraffic")

obs = e.reset()
while True:
    action, states = model.predict(obs)
    print("action: ", actions[action])
    obs, reward, done, info = e.step(action)
    if done:
        break

e.close()
