import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('log.csv', names=['reward'])

cumsum, moving_aves = [0], []

N=1000

for i, x in enumerate(df['reward'], 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves.append(moving_ave)

plt.plot(moving_aves, label="reward (N=1000)", linewidth = 0.1)
plt.title("YamaXEnv")
plt.xlabel("timestep")
plt.ylabel("reward")
plt.legend(loc='upper left')
plt.show()
