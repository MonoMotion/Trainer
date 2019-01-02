from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import sys
import flom

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('input', type=str, help='Input motion file')
parser.add_argument('--fps', type=float, default=0.01)
parser.add_argument('-o', '--output', type=str, help='output plot file', required=True)
args = parser.parse_args()

motion = flom.load(args.input)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

joints = [j for j in motion.joint_names() if not j.startswith('ignore_')]

def plot_frames(motion):
    motion.set_loop(flom.LoopType.None_)
    x = [i * args.fps for i, frame in enumerate(motion.frames(args.fps))]
    ys = [frame.positions for frame in motion.frames(args.fps)]
    for joint in joints:
        ax.plot(x, [y[joint] for y in ys], label=joint)

plot_frames(motion)

ax.legend()
plt.savefig(args.output)
