from argparse import ArgumentParser
import flom

from trainer import train, preview


def make_parser():
    parser = ArgumentParser(description='Train the motion to fit to effectors')
    parser.add_argument('-i', '--input', type=str, help='Input motion file', required=True)
    parser.add_argument('-r', '--robot', type=str, help='Input robot model file', required=True)
    parser.add_argument('-t', '--timestep', type=float, help='Timestep', default=0.0165/8)
    parser.add_argument('-s', '--frame-skip', type=int, help='Frame skip', default=8)

    return parser

def main(args):
    motion = flom.load(args.input)
    trained = train(motion, args.robot, args.timestep, args.frame_skip)
    preview(trained, args.robot)

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)
