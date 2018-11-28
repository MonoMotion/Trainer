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

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

frames = data['frames']
joints = frames[0]['position'].keys()

x = [frame['timepoint'] for frame in frames]
ys = [frame['position'] for frame in frames]
for joint in joints:
    ax.plot(x, [y[joint] for y in ys], label=joint)

ax.legend()
plt.savefig(args.output)
