import math

from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene

from .utils import dictzip

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

