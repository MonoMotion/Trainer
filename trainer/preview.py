from .simulation import create_scene, reset, apply_joints, render

def preview(motion, robot_file, timestep=0.0165/8, frame_skip=8):
    scene = create_scene(timestep, frame_skip)

    robot, parts, joints = reset(scene, robot_file)

    while True:
        scene.global_step()

        frame = motion.frame_at(scene.cpp_world.ts)
        apply_joints(joints, frame.positions)

        render(scene)
