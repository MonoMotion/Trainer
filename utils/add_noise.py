from argparse import ArgumentParser
import sys
import random

import flom

parser = ArgumentParser(description='Add noise to motion file')
parser.add_argument('input', type=str, help='Input motion file')
parser.add_argument('-r', '--random', type=float, default=-0.1,
                    help='Randomness')
parser.add_argument('-o', '--output', type=str, help='output plot file', required=True)
args = parser.parse_args()

def add_noise(motion):
    for t, frame in motion.keyframes():
        new_frame = frame.get()
        positions = {
            k: v + random.random() * args.random
            for k, v in new_frame.positions.items()
        }
        new_frame.positions = positions
        frame.set(new_frame)

motion = flom.load(args.input)

add_noise(motion)

motion.dump(args.output)
