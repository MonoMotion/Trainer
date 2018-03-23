import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

dn1 = sys.argv[1]
dn2 = sys.argv[2]

monitor1 = pd.read_csv(os.path.join(dn1, "monitor/log.csv"), names=['time_elapsed', 'reward_sum', 'final_reward', 'maximum_leg_error', 'num_timesteps', 'distance_sum', 'final_distance', 'unperm_sum'])
monitor2 = pd.read_csv(os.path.join(dn2, "monitor/log.csv"), names=['time_elapsed', 'reward_sum', 'final_reward', 'maximum_leg_error', 'num_timesteps', 'distance_sum', 'final_distance', 'unperm_sum'])

entry = sys.argv[3]
N1 = sys.argv[4]
N2 = sys.argv[5]

data1 = monitor1[entry]
data2 = monitor2[entry]

if len(data1) > len(data2):
    data1 = data1[:len(data2)]
elif len(data2) > len(data1):
    data2 = data2[:len(data1)]

def calcmovingaves(ary, fn):
    n = int(fn)
    cumsum, moving_aves = [0], []

    for i, x in enumerate(ary, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=n:
            moving_ave = (cumsum[i] - cumsum[i-n])/n
            moving_aves.append(moving_ave)
    return moving_aves

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(data1, label=dn1, linewidth=0.5, color='red', alpha=0.1)
ax1.plot(calcmovingaves(data1, N1), label=(dn1 + " (N={})".format(N1)), linewidth=0.5, color='red', alpha=0.3)
ax1.plot(calcmovingaves(data1, N2), label=(dn1 + " (N={})".format(N2)), linewidth=0.5, color='red', alpha=0.8)
ax1.plot(data2, label=dn2, linewidth = 0.5, color='blue', alpha=0.1)
ax1.plot(calcmovingaves(data2, N1), label=(dn2 + " (N={})".format(N1)), linewidth=0.5, color='blue', alpha=0.3)
ax1.plot(calcmovingaves(data2, N2), label=(dn2 + " (N={})".format(N2)), linewidth=0.5, color='blue', alpha=0.8)

h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc='upper left')

ax1.set_title("YamaXEnv - {} ({}, {})".format(entry, dn1, dn2))
ax1.set_xlabel("episode")
ax1.set_ylabel(entry)
ax1.grid(True)
plt.savefig('{}-{}+{}_moveavg{}-{}.png'.format(entry, dn1, dn2, N1, N2))
plt.show()
