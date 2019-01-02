from argparse import ArgumentParser
import math
import numpy as np
import quaternion
from functools import reduce
from operator import mul

from evostra import EvolutionStrategy
from preview_motion import create_motion_iterator, create_scene, reset, apply_joints, render

import flom

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('-i', '--input', type=str, help='Input motion file', required=True)
parser.add_argument('-r', '--robot', type=str, help='Input robot model file', required=True)
parser.add_argument('-t', '--timestep', type=float, help='Timestep', default=0.0165/8)
parser.add_argument('-s', '--frame-skip', type=int, help='Frame skip', default=8)

def apply_weights(positions, weights):
    # sort is required because frame order is nondeterministic
    return {k: v + w for w, (k, v) in zip(weights, sorted(positions.items()))}

def calc_reward(weight, robot, frame, parts):
    robot.query_position()
    diff = 0
    for name, effector in frame.effectors.items():
        pose = parts[name].pose() if name != 'base_link' else robot.root_part.pose()
        if effector.location:
            diff += np.linalg.norm(pose.xyz() - effector.location.vec) ** 2 * effector.location.weight
        if effector.rotation:
            quat1 = np.quaternion(*effector.rotation.quat)
            quat2 = np.quaternion(*pose.quatertion())
            diff += quaternion.rotation_intrinsic_distance(quat1, quat2) ** 2 * effector.rotation.weight
    k = 1
    normalized = k * diff / len(frame.effectors)
    return - math.exp(normalized) + 1


def main(args):
    scene = create_scene(args.timestep, args.frame_skip)
    motion = create_motion_iterator(args.input)
    robot, parts, joints = reset(scene, args.robot)
    robot.query_position()
    _, _, initial_z = robot.root_part.pose().xyz()

    def step(weights, enable_render=False):
        robot, parts, joints = reset(scene, args.robot)

        reward_sum = 0
        for frame_weight in weights:
            scene.global_step()

            frame = motion.frame_at(scene.cpp_world.ts)

            reward_sum += calc_reward(frame_weight, robot, frame, parts)

            apply_joints(joints, apply_weights(frame.positions, frame_weight))

            if enable_render:
                render(scene)

        return reward_sum

    weights = np.zeros(shape=(321, 10), dtype='float')
    es = EvolutionStrategy(weights, step, population_size=20, sigma=0.1, learning_rate=0.03, decay=0.995, num_threads=1)
    es.run(1000, print_step=10)
    while True:
        step(es.get_weights(), enable_render=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
