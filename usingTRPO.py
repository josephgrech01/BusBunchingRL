from sb3_contrib import TRPO

from env import SumoEnv

actions = ['Hold', 'Skip', 'Proceed']
e = SumoEnv(gui=True, noWarnings=True, epLen=500, traffic=10, bunched=True)

# no traffic
# model = TRPO.load("models/trpoNoTraffic")

# traffic
model = TRPO.load("models/trpoTraffic")

obs = e.reset()
while True:
    action, states = model.predict(obs, deterministic=True)
    print("action: ", actions[action])
    obs, reward, done, info = e.step(action)
    if done:
      break

e.close()