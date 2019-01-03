from .simulation import create_scene, reset, apply_joints, render

EFFECTOR_SPHERE_RADIUS_RATIO = 0.05
EFFECTOR_SPHERE_COLOR = 0x2ecc77

def create_effector_visualizer(scene, effectors):
    def create(eff):
        if eff.location:
            x, y, z = eff.location.vec
            radius = eff.location.weight * EFFECTOR_SPHERE_RADIUS_RATIO
            return scene.cpp_world.debug_sphere(x, y, z, radius, EFFECTOR_SPHERE_COLOR)

    return {name: create(eff) for name, eff in effectors.items()}

def preview(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = create_scene(timestep, frame_skip)

    robot, parts, joints = reset(scene, robot_file)

    effector_vis = None
    while True:
        scene.global_step()

        frame = motion.frame_at(scene.cpp_world.ts)
        apply_joints(joints, frame.positions)
        effector_vis = create_effector_visualizer(scene, frame.effectors)

        render(scene)
