import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot(logfile, entry1, entry2, N1, N2, title=None, filename=None):
    monitor = pd.read_csv(logfile, names=['time_elapsed', 'reward_sum', 'final_reward', 'maximum_leg_error', 'num_timesteps', 'distance_sum', 'final_distance', 'unperm_sum'])

    data1 = monitor[entry1]
    data2 = monitor[entry2]

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

    ax1.plot(data1, label=entry1, linewidth=0.5, color='red', alpha=0.1)
    ax1.plot(calcmovingaves(data1, N1), label=(entry1 + " (N={})".format(N1)), linewidth=0.5, color='red', alpha=0.3)
    ax1.plot(calcmovingaves(data1, N2), label=(entry1 + " (N={})".format(N2)), linewidth=0.5, color='red', alpha=0.8)
    ax2 = ax1.twinx()
    ax2.plot(data2, label=entry2, linewidth=0.5, color='blue', alpha=0.1)
    ax2.plot(calcmovingaves(data2, N1), label=(entry2 + " (N={})".format(N1)), linewidth=0.5, color='blue', alpha=0.3)
    ax2.plot(calcmovingaves(data2, N2), label=(entry2 + " (N={})".format(N2)), linewidth=0.5, color='blue', alpha=0.8)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')

    title = title or "YamaXEnv - {} ({}, {})".format(os.getcwd().split('/')[-1], entry1, entry2)
    ax1.set_title(title)
    ax1.set_xlabel("episode")
    ax1.set_ylabel(entry1)
    ax2.set_ylabel(entry2)
    ax1.grid(True)
    filename = filename or '{}+{}_{}-{}.png'.format(entry1, entry2, N1, N2)
    plt.savefig(filename)

if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    plt.show()
