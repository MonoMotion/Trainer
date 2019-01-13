from .simulation import reset, apply_joints
from .utils import select_location
from .silver_bullet import Color, Scene
from pybullet_utils import bullet_client
import pybullet

import math

EFFECTOR_SPHERE_SIZE_RATIO = 2.5
EFFECTOR_SPHERE_COLOR_RATIO = 1000

def create_effector_marker(scene, motion, robot, effectors, pre):
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
            size = motion.effector_weight(name).location * EFFECTOR_SPHERE_SIZE_RATIO
            x, y, z = target
            return scene.draw_text(name, [x, y, z], size=size, color=color, replace=pre[name] if pre else None)

    return {name: create(name, eff) for name, eff in effectors.items()}

def preview(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    gui_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    scene = Scene(timestep, frame_skip, client=gui_client)

    robot = reset(scene, robot_file)

    effector_marks = None
    c = 0
    while True:
        scene.step()

        frame = motion.frame_at(scene.ts)
        apply_joints(robot, frame.positions)
        if c % 10 == 0:
            effector_marks = create_effector_marker(scene, motion, robot, frame.effectors, effector_marks)
        c += 1
