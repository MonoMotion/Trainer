from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('input', type=str, help='Input motion file')
parser.add_argument('-o', '--output', type=str, help='output plot file', required=True)
args = parser.parse_args()

data = np.loadtxt(args.input, delimiter=',')
num_joints = int(data.T[1].max())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for i in range(num_joints):
    x, y = np.array([[d[0], d[2]] for d in data if int(d[1] == i)]).T
    assert len(x) == len(y)
    ax.plot(x, y, label=str(i))

ax.legend()
plt.savefig(args.output)
