import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

nc = pd.read_csv('results/final/csvs/noControl/noTraffic.csv')
rbc = pd.read_csv('results/final/csvs/rbc/noTraffic.csv')
trpo = pd.read_csv('results/final/csvs/trpo/noTraffic.csv')
ppo = pd.read_csv('results/final/csvs/ppo/noTraffic.csv')


ncTime = nc['meanWaitTime'].tolist()
rbcTime = rbc['meanWaitTime'].tolist()
trpoTime = trpo['meanWaitTime'].tolist()
ppoTime = ppo['meanWaitTime'].tolist()

ncSD = nc['headwaySD'].tolist()
rbcSD = rbc['headwaySD'].tolist()
trpoSD = trpo['headwaySD'].tolist()
ppoSD = ppo['headwaySD'].tolist()

ncDisp = nc['dispersion'].tolist()
rbcDisp = rbc['dispersion'].tolist()
trpoDisp = trpo['dispersion'].tolist()
ppoDisp = ppo['dispersion'].tolist()

# # Mean Waiting Time
fig, ax1 = plt.subplots(1, 1)
ax1.set_xlabel('RL Step')
ax1.set_ylabel('Mean waiting time (mins)')
ax1.set_title('Mean Waiting Time, No Traffic')
# values are scaled back to reality and converted to minutes
ax1.plot(range(1, len(ncTime) + 1), [(mean*9)/60 for mean in ncTime], color='blue', linestyle='-', linewidth=1.5, label='No Control')
ax1.plot(range(1, len(rbcTime) + 1), [(mean*9)/60 for mean in rbcTime], color='red', linestyle='-', linewidth=1.5, label='Rule-Based Control')
ax1.plot(range(1, len(trpoTime) + 1), [(mean*9)/60 for mean in trpoTime], color='green', linestyle='-', linewidth=1.5, label='TRPO')
ax1.plot(range(1, len(ppoTime) + 1), [(mean*9)/60 for mean in ppoTime], color='black', linestyle='-', linewidth=1.5, label='PPO')
ax1.grid()
plt.legend()
plt.savefig('results/final/waitTime/noTraffic.jpg')
plt.show()
plt.clf()

# # Headway Standard Deviation
fig, ax1 = plt.subplots(1, 1)
ax1.set_xlabel('RL Step')
ax1.set_ylabel('Headway Standard Deviation')
ax1.set_title('Headway Standard Deviation, No Traffic')
ax1.plot(range(1, len(ncSD) + 1), ncSD, color='blue', linestyle='-', linewidth=1.5, label='No Control')
ax1.plot(range(1, len(rbcSD) + 1), rbcSD, color='red', linestyle='-', linewidth=1.5, label='Rule-Based Control')
ax1.plot(range(1, len(trpoSD) + 1), trpoSD, color='green', linestyle='-', linewidth=1.5, label='TRPO')
ax1.plot(range(1, len(ppoSD) + 1), ppoSD, color='black', linestyle='-', linewidth=1.5, label='PPO')
ax1.grid()
plt.legend()
plt.savefig('results/final/headwaySD/noTraffic.jpg')
plt.show()
plt.clf()

# # Occupancy Dispersion
fig, ax1 = plt.subplots(1, 1)
ax1.set_xlabel('RL Step')
ax1.set_ylabel('Occupancy Dispersion')
ax1.set_title('Occupancy Dispersion, No Traffic')
ax1.plot(range(1, len(ncDisp) + 1), ncDisp, color='blue', linestyle='-', linewidth=1.5, label='No Control')
ax1.plot(range(1, len(rbcDisp) + 1), rbcDisp, color='red', linestyle='-', linewidth=1.5, label='Rule-Based Control')
ax1.plot(range(1, len(trpoDisp) + 1), trpoDisp, color='green', linestyle='-', linewidth=1.5, label='TRPO')
ax1.plot(range(1, len(ppoDisp) + 1), ppoDisp, color='black', linestyle='-', linewidth=1.5, label='PPO')
ax1.grid()
plt.legend()
plt.savefig('results/final/occupancyDispersion/noTraffic.jpg')
plt.show()
plt.clf()

x = ['Rule-Based Control', 'TRPO', 'PPO']

# No Traffic
holdNoTraffic = np.array([9, 30.5, 56.7])
skipNoTraffic = np.array([8.2, 0, 0])
proceedNoTraffic = np.array([82.8, 69.5, 43.3])

actions = {'Hold': holdNoTraffic, 'Skip': skipNoTraffic, 'Proceed': proceedNoTraffic}

fig, ax = plt.subplots()
bottom = np.zeros(3)

for a, action in actions.items():
    p = ax.bar(x, action, label=a, bottom=bottom)
    bottom += action
    ac = [str(x)+'%' if x != 0 else '' for x in action]
    ax.bar_label(p, labels=ac, label_type='center')

ax.set_title('Distribution of Actions - No Traffic')
ax.legend()
plt.savefig('results/final/actions/noTraffic.jpg')
plt.show()
plt.clf()

# Traffic, Evenly spaced
holdNoTraffic = np.array([41.5, 48.3, 77])
skipNoTraffic = np.array([20.6, 1.6, 11.8])
proceedNoTraffic = np.array([37.9, 50.1, 11.2])

actions = {'Hold': holdNoTraffic, 'Skip': skipNoTraffic, 'Proceed': proceedNoTraffic}

fig, ax = plt.subplots()
bottom = np.zeros(3)

for a, action in actions.items():
    p = ax.bar(x, action, label=a, bottom=bottom)
    bottom += action
    ac = [str(x)+'%' if x != 0 else '' for x in action]
    ax.bar_label(p, labels=ac, label_type='center')

ax.set_title('Distribution of Actions - Traffic, Evenly Spaced')
ax.legend()
plt.savefig('results/final/actions/Traffic.jpg')
plt.show()
plt.clf()

# Traffic, Bunched
holdNoTraffic = np.array([40.9, 51.5, 64.5])
skipNoTraffic = np.array([22.2, 6.8, 22.8])
proceedNoTraffic = np.array([36.9, 41.7, 12.8])

actions = {'Hold': holdNoTraffic, 'Skip': skipNoTraffic, 'Proceed': proceedNoTraffic}

fig, ax = plt.subplots()
bottom = np.zeros(3)

for a, action in actions.items():
    p = ax.bar(x, action, label=a, bottom=bottom)
    bottom += action
    ac = [str(x)+'%' if x != 0 else '' for x in action]
    ax.bar_label(p, labels=ac, label_type='center')

ax.set_title('Distribution of Actions - Traffic, Bunched')
ax.legend()
plt.savefig('results/final/actions/TrafficBunched.jpg')
plt.show()
plt.clf()