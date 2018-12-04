# TODO: Remove this after organizing file structure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from argparse import ArgumentParser
import json
import math

from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene

from gym_yamax.motion import MotionIterator, get_frame_at
from gym_yamax.utils import dictzip

parser = ArgumentParser(description='Plot motion file')
parser.add_argument('-i', '--input', type=str, help='Input motion file', required=True)
parser.add_argument('-r', '--robot', type=str, help='Input robot model file', required=True)
parser.add_argument('-t', '--timestep', type=float, help='Timestep', default=0.0165/8)
parser.add_argument('-s', '--frame-skip', type=int, help='Frame skip', default=8)

def create_motion_iterator(path):
    with open(path) as f:
        data = json.load(f)

    return MotionIterator(data)

def create_scene(ts, skip):
    return SinglePlayerStadiumScene(gravity=9.8, timestep=ts, frame_skip=skip)

def load_urdf(scene, path, with_self_collision=True):
    pose = cpp_household.Pose()
    urdf = scene.cpp_world.load_urdf(
        path,
        pose,
        False,  # fixed_base
        with_self_collision)

    parts = {part.name: part for part in urdf.parts}

    def configure_joint(j):
        if j.name.startswith('ignore'):
            j.set_motor_torque(0)
        else:
            j.power_coef, j.max_velocity = j.limits()[2:4]
        return j

    joints = {j.name: configure_joint(j) for j in urdf.joints}

    camera = scene.cpp_world.new_camera_free_float(320, 240, "video_camera")

    return urdf, parts, joints, camera

def render(scene):
    scene.human_render_detected = True
    return scene.cpp_world.test_window()

def reset_position(cpp_robot, parts):
    cpp_robot.query_position()

    initial_z = - min(p.pose().xyz()[2] for p in parts.values()) + 0.01

    cpose = cpp_household.Pose()
    cpose.set_xyz(0, 0, initial_z)
    cpose.set_rpy(0, 0, 0)
    cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)

def reset(scene, path):
    scene.episode_restart()

    robot, parts, joints, _ = load_urdf(scene, path)

    reset_position(robot, parts)

    return robot, parts, joints

def apply_joints(joints, positions):
    for _, (j, pos) in dictzip(joints, positions):
        j.set_servo_target(pos, 0.1, 1.0, 100000)

def main(args):
    scene = create_scene(args.timestep, args.frame_skip)
    motion = create_motion_iterator(args.input)

    robot, parts, joints = reset(scene, args.robot)

    while True:
        scene.global_step()

        frame = get_frame_at(scene.cpp_world.ts, motion)
        apply_joints(joints, frame)

        render(scene)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
