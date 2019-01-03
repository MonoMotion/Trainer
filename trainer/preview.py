from .simulation import create_scene, reset, apply_joints, render

EFFECTOR_SPHERE_RADIUS = 0.05
EFFECTOR_SPHERE_COLOR = 0x2ecc77

def create_effectors(scene, parts, motion):
    def create(name):
        x, y, z = parts[name].pose().xyz()
        return scene.cpp_world.debug_sphere(x, y, z, EFFECTOR_SPHERE_RADIUS, EFFECTOR_SPHERE_COLOR)

    return {name: create(name) for name in motion.effector_names()}

def preview(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = create_scene(timestep, frame_skip)

    robot, parts, joints = reset(scene, robot_file)
    effectors = create_effectors(scene, parts, motion)

    while True:
        scene.global_step()

        frame = motion.frame_at(scene.cpp_world.ts)
        apply_joints(joints, frame.positions)

        render(scene)
