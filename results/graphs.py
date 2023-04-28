import pandas as pd
import matplotlib.pyplot as plt

nc = pd.read_csv('results/final/csvs/noControl/noTraffic.csv')
rbc = pd.read_csv('results/final/csvs/rbc/noTraffic.csv')
trpo = pd.read_csv('results/final/csvs/trpo/noTraffic.csv')
ppo = pd.read_csv('results/final/csvs/ppo/noTraffic.csv')


# dfSD = pd.read_csv('csvs/.csv')

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

fig, ax1 = plt.subplots(1, 1)
ax1.set_xlabel('Step')
ax1.set_ylabel('Mean waiting time (mins)')
ax1.set_title('Mean Waiting Time, No Traffic')
ax1.plot(range(1, len(ncTime) + 1), [(mean*9)/60 for mean in ncTime], color='blue', linestyle='-', linewidth=1.5, label='No Control')
ax1.plot(range(1, len(rbcTime) + 1), [(mean*9)/60 for mean in rbcTime], color='red', linestyle='-', linewidth=1.5, label='Rule-Based Control')
ax1.plot(range(1, len(trpoTime) + 1), [(mean*9)/60 for mean in trpoTime], color='green', linestyle='-', linewidth=1.5, label='TRPO')
ax1.plot(range(1, len(ppoTime) + 1), [(mean*9)/60 for mean in ppoTime], color='black', linestyle='-', linewidth=1.5, label='PPO')
ax1.grid()
plt.legend()
plt.savefig('results/final/waitTime/noTraffic.jpg')
plt.show()
plt.clf()


# fig, ax1 = plt.subplots(1, 1)
# ax1.set_xlabel('Step')
# ax1.set_ylabel('Headway Standard Deviation')
# ax1.set_title('Headway Standard Deviation, Traffic, Bunched')
# ax1.plot(range(1, len(ncVals) + 1), ncVals, color='blue', linestyle='-', linewidth=1.5, label='No Control')
# ax1.plot(range(1, len(ppoVals) + 1), ppoVals, color='red', linestyle='-', linewidth=1.5, label='PPO')
# # ax1.plot(range(1, len(trpoVals) + 1), trpoVals, color='green', linestyle='-', linewidth=1.5, label='TRPO')
# ax1.grid()
# plt.legend()
# plt.savefig('results/test/headwaySDTrafficBunched.jpg')
# plt.show()
# plt.clf()