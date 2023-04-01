import pandas as pd
import matplotlib.pyplot as plt

# NCTraffic = pd.read_csv('results/csvs/noControlTraffic.csv')
PPOTrafficBunched = pd.read_csv('results/csvs/PPOSDTrafficBunched.csv')
TRPOTrafficBunched = pd.read_csv('results/csvs/TRPOSDTrafficBunched.csv')

# dfSD = pd.read_csv('csvs/.csv')

# ncVals = NCTraffic['meanWaitTime'].tolist()
ppoVals = PPOTrafficBunched['headwaySD'].tolist()
trpoVals = TRPOTrafficBunched['headwaySD'].tolist()

# fig, ax1 = plt.subplots(1, 1)
# ax1.set_xlabel('Step')
# ax1.set_ylabel('Mean waiting time (mins)')
# ax1.set_title('Mean Waiting Time, Traffic, Bunched')
# # ax1.plot(range(1, len(ncVals) + 1), [(mean*9)/60 for mean in ncVals], color='blue', linestyle='-', linewidth=3, label='No Control')
# ax1.plot(range(1, len(ppoVals) + 1), [(mean*9)/60 for mean in ppoVals], color='red', linestyle='-', linewidth=3, label='PPO')
# ax1.plot(range(1, len(trpoVals) + 1), [(mean*9)/60 for mean in trpoVals], color='green', linestyle='-', linewidth=3, label='TRPO')
# ax1.grid()
# plt.legend()
# plt.savefig('results/waitingTimeTrafficBunched.jpg')
# plt.show()
# plt.clf()


fig, ax1 = plt.subplots(1, 1)
ax1.set_xlabel('Step')
ax1.set_ylabel('Headway Standard Deviation')
ax1.set_title('Headway Standard Deviation, Traffic, Bunched')
ax1.plot(range(1, len(ppoVals) + 1), ppoVals, color='red', linestyle='-', linewidth=2, label='PPO')
ax1.plot(range(1, len(trpoVals) + 1), trpoVals, color='green', linestyle='-', linewidth=2, label='TRPO')
ax1.grid()
plt.legend()
plt.savefig('results/headwayStandardDeviation.jpg')
plt.show()
plt.clf()