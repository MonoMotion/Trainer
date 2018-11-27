from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import json

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('input', type=str, help='Input motion file')
parser.add_argument('-o', '--output', type=str, help='output plot file', required=True)
args = parser.parse_args()

with open(args.input) as f:
    data = json.load(f)
num_joints = data['num_joints']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

seq = data['sequence']

x = list(map(float, seq.keys()))
ys = np.array(list(seq.values())).T
for (i, y) in enumerate(ys):
    ax.plot(x, y, label=str(i))

ax.legend()
plt.savefig(args.output)
