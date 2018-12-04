from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('input', type=str, help='Input motion file')
parser.add_argument('-l', '--loop', type=int, default=2,
                    help='The number of repetitions when loop == wrap')
parser.add_argument('-o', '--output', type=str, help='output plot file', required=True)
args = parser.parse_args()

with open(args.input) as f:
    data = json.load(f)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

loop = data['loop']
frames = data['frames']
joints = [j for j in frames[0]['position'].keys() if not j.startswith('ignore_')]


def plot_frames(frames):
    x = [frame['timepoint'] for frame in frames]
    ys = [frame['position'] for frame in frames]
    for joint in joints:
        ax.plot(x, [y[joint] for y in ys], label=joint)


if loop == 'none':
    plot_frames(frames)
elif loop == 'wrap':
    looped = frames
    for _ in range(args.loop - 1):
        looped = looped + [{'timepoint': frame['timepoint'] + looped[-1]
                            ['timepoint'], 'position': frame['position']} for frame in frames]
    plot_frames(looped)
else:
    sys.exit("invalid loop value: {}".format(loop))

ax.legend()
plt.savefig(args.output)
