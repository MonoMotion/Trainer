from .simulation import create_scene, reset, apply_joints, render

import flom

import math

EFFECTOR_SPHERE_RADIUS_RATIO = 0.05
EFFECTOR_SPHERE_COLOR_RATIO = 1000

def create_effector_marker(scene, motion, robot, parts, effectors):
    def calc_color(differ):
        r =  - math.exp(-differ * EFFECTOR_SPHERE_COLOR_RATIO) + 1
        color_red = int(0xff * r)
        color_blue = int(0xff + 1 - r)
        color = color_blue | color_red * (0xFFFF + 1)
        return color

    def select_pose(ty, pose, root):
        if ty == flom.CoordinateSystem.World:
            return pose
        elif ty == flom.CoordinateSystem.Local:
            return pose + root
        else:
            assert False  # unreachable

    def create(name, eff):
        part = parts[name]
        ty = motion.effector_type(name)
        if eff.location:
            current = part.pose().xyz()
            target = select_pose(ty.location, eff.location.vec, robot.root_part.pose().xyz())
            differ = sum((c - t) ** 2 for c, t in zip(current, target)) / 3

            color = calc_color(differ)
            radius = eff.location.weight * EFFECTOR_SPHERE_RADIUS_RATIO
            x, y, z = target
            return scene.cpp_world.debug_sphere(x, y, z, radius, color)

    robot.query_position()
    return {name: create(name, eff) for name, eff in effectors.items()}

def preview(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = create_scene(timestep, frame_skip)

    robot, parts, joints = reset(scene, robot_file)

    while True:
        scene.global_step()

        frame = motion.frame_at(scene.cpp_world.ts)
        apply_joints(joints, frame.positions)
        effector_marks = create_effector_marker(scene, motion, robot, parts, frame.effectors)

        render(scene)
