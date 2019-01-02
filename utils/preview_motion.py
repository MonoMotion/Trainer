# TODO: Remove this after organizing file structure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from argparse import ArgumentParser
import flom
from trainer import preview

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('-i', '--input', type=str, help='Input motion file', required=True)
parser.add_argument('-r', '--robot', type=str, help='Input robot model file', required=True)
parser.add_argument('-t', '--timestep', type=float, help='Timestep', default=0.0165/8)
parser.add_argument('-s', '--frame-skip', type=int, help='Frame skip', default=8)

def main(args):
    motion = flom.load(args.input)
    preview(motion, args.robot, args.timestep, args.frame_skip)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
