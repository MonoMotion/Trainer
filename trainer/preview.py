from .simulation import create_scene, reset, apply_joints, render

import math

EFFECTOR_SPHERE_RADIUS_RATIO = 0.05
EFFECTOR_SPHERE_COLOR_RATIO = 1000

def create_effector_visualizer(scene, parts, effectors):
    def calc_color(differ):
        r =  - math.exp(-differ * EFFECTOR_SPHERE_COLOR_RATIO) + 1
        color_red = int(0xff * r)
        color_blue = int(0xff + 1 - r)
        color = color_blue | color_red * (0xFFFF + 1)
        return color

    def create(part, eff):
        if eff.location:
            current = part.pose().xyz()
            target = eff.location.vec
            differ = sum((c - t) ** 2 for c, t in zip(current, target)) / 3

            color = calc_color(differ)
            radius = eff.location.weight * EFFECTOR_SPHERE_RADIUS_RATIO
            x, y, z = target
            return scene.cpp_world.debug_sphere(x, y, z, radius, color)

    return {name: create(parts[name], eff) for name, eff in effectors.items()}

def preview(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = create_scene(timestep, frame_skip)

    robot, parts, joints = reset(scene, robot_file)

    effector_vis = None
    while True:
        scene.global_step()

        frame = motion.frame_at(scene.cpp_world.ts)
        apply_joints(joints, frame.positions)
        effector_vis = create_effector_visualizer(scene, parts, frame.effectors)

        render(scene)
