from argparse import ArgumentParser
import sys
import json
import random

parser = ArgumentParser(description='Add noise to motion file')
parser.add_argument('input', type=str, help='Input motion file')
parser.add_argument('-r', '--random', type=float, default=-0.1,
                    help='Randomness')
parser.add_argument('-o', '--output', type=str, help='output plot file', required=True)
args = parser.parse_args()

def add_noise(frame):
    position = {
        k: v + random.random() * args.random
        for k, v in frame['position'].items()
    }

    return {
            'timepoint': frame['timepoint'],
            'position': position
            }

with open(args.input) as f:
    data = json.load(f)

frames = data['frames']

with_noise = [add_noise(frame) for frame in frames]

data['frames'] = with_noise

with open(args.output, 'w') as f:
    json.dump(data, f, indent=2)
