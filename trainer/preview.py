from .simulation import create_scene, reset, apply_joints
from .utils import select_location
from .silver_bullet import Color

import flom

import math

EFFECTOR_SPHERE_RADIUS_RATIO = 0.05
EFFECTOR_SPHERE_COLOR_RATIO = 1000

def create_effector_marker(scene, motion, robot, effectors):
    def calc_color(diff):
        r =  - math.exp(-diff * EFFECTOR_SPHERE_COLOR_RATIO) + 1
        return Color(r, 0, 1 - r)

    def create(name, eff):
        ty = motion.effector_type(name)
        if eff.location:
            current = robot.link_state(name).pose.vector
            root_pose = robot.link_state(robot.root_link).pose
            target = select_location(ty.location, eff.location.vector, root_pose)
            differ = sum((c - t) ** 2 for c, t in zip(current, target)) / 3

            color = calc_color(differ)
            radius = motion.effector_weight(name).location * EFFECTOR_SPHERE_RADIUS_RATIO
            x, y, z = target
            return scene.draw_sphere([x, y, z], radius=radius, color=color)

    return {name: create(name, eff) for name, eff in effectors.items()}

def delete_effector_marker(scene, effector_marks):
    for _, m in effector_marks.items():
       scene.remove_debug_object(m)

def preview(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = create_scene(timestep, frame_skip)

    robot = reset(scene, robot_file)

    effector_marks = None
    while True:
        scene.step()

        frame = motion.frame_at(scene.ts)
        apply_joints(robot, frame.positions)
        if effector_marks:
            delete_effector_marker(scene, effector_marks)
        effector_marks = create_effector_marker(scene, motion, robot, frame.effectors)
