from argparse import ArgumentParser
import math
import numpy as np
from functools import reduce
from operator import mul

from evostra import EvolutionStrategy
from preview_motion import create_motion_iterator, create_scene, reset, get_frame_at, apply_joints, render

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('-i', '--input', type=str, help='Input motion file', required=True)
parser.add_argument('-r', '--robot', type=str, help='Input robot model file', required=True)
parser.add_argument('-t', '--timestep', type=float, help='Timestep', default=0.0165/8)
parser.add_argument('-s', '--frame-skip', type=int, help='Frame skip', default=8)

def apply_weights(frame, weights):
    # sort is required because frame order is nondeterministic
    return {k: v + w for w, (k, v) in zip(weights, sorted(frame.items()))}

def calc_reward(robot, initial_z, last_x):
    robot.query_position()
    x, y, z = robot.root_part.pose().xyz()
    euler = robot.root_part.pose().rpy()
    c = [math.cos(a / 2) for a in euler]
    s = [math.sin(a / 2) for a in euler]
    axis_angle = 2 * math.acos(reduce(mul, c) - reduce(mul, s))
    return - (x - last_x) - abs(z - initial_z), x, axis_angle > math.pi / 4

def main(args):
    scene = create_scene(args.timestep, args.frame_skip)
    motion = create_motion_iterator(args.input)
    robot, parts, joints = reset(scene, args.robot)
    robot.query_position()
    _, _, initial_z = robot.root_part.pose().xyz()

    def step(weights, enable_render=False):
        robot, parts, joints = reset(scene, args.robot)

        reward_sum = 0
        last_x = 0
        for frame_weight in weights:
            scene.global_step()

            reward, last_x, done = calc_reward(robot, initial_z, last_x)
            if done:
                return - 0.2 * last_x
            reward_sum += reward

            frame = get_frame_at(scene.cpp_world.ts, motion)
            apply_joints(joints, apply_weights(frame, frame_weight))

            if enable_render:
                render(scene)

        return reward_sum

    weights = np.zeros(shape=(321, 10), dtype='float')
    es = EvolutionStrategy(weights, step, population_size=20, sigma=0.1, learning_rate=0.03, decay=0.995, num_threads=1)
    es.run(1000, print_step=10)
    step(es.get_weights(), enable_render=True)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
