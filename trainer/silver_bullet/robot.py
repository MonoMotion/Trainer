import dataclasses
import pybullet
from pybullet_utils import bullet_client
import numpy as np
from .scene import Scene

import functools
from typing import Optional, Dict

@dataclasses.dataclass
class Pose:
    vector: np.ndarray
    quaternion: np.quaternion

@dataclasses.dataclass
class JointState:
    position: float
    velocity: float

@dataclasses.dataclass
class LinkState:
    pose: Pose
    linear_velocity: Optional[np.ndarray]
    angular_velocity: Optional[np.ndarray]

@dataclasses.dataclass
class ClientWithBody:
    client: bullet_client.BulletClient
    body_id: int

    def __getattr__(self, name):
        method = getattr(self.client, name)
        return functools.partial(method, bodyUniqueId=self.body_id)

@dataclasses.dataclass
class Robot:
    body_id: int
    scene: dataclasses.InitVar[Scene]

    joints: Dict[str, int] = dataclasses.field(init=False)
    links: Dict[str, int] = dataclasses.field(init=False)
    client: ClientWithBody = dataclasses.field(init=False)

    def __post_init__(self, scene):
        self.client = ClientWithBody(scene.client, self.body_id)
        nr_joints = self.client.getNumJoints()
        self.links = {}
        self.joints = {}
        for idx in range(nr_joints):
            result = self.client.getJointInfo(jointIndex=idx)
            joint_name = result[1]
            link_name = result[12]
            link_idx = result[16]
            self.links[link_name] = link_idx
            self.joints[joint_name] = idx

    def joint_state(self, name: str) -> JointState:
        joint_id = self.joints[name]
        pos, vel, _, _ = self.client.getJointState(jointIndex=joint_id)
        return JointState(pos, vel)

    def link_state(self, name: str, compute_velocity=False) -> LinkState:
        link_id = self.links[name]
        if link_id == -1:
            pos, ori = self.client.getBasePositionAndOrientation()
            if compute_velocity:
                a_vel, l_vel = self.client.getBaseVelocity()
            else:
                a_vel = None
                l_vel = None
        else:
            pos, ori, _, _, l_vel, a_vel = self.client.getLinkState(linkIndex=link_id, computeLinkVelocity=compute_velocity)
        pose = Pose(np.array(pos), np.quaternion(*ori))
        return LinkState(pose, l_vel, a_vel)

    def bring_on_the_ground(self, padding: float = 0):
        h = min(self.link_state(name).pose.vector[2] for name in self.links.keys())
        if h > 0:
            raise RuntimeError("Robot is already on the ground")
        self.client.resetBasePositionAndOrientation(posObj=[0, 0, -h + padding], ornObj=[0, 0, 0, 1])

def load_urdf(scene, path, flags=pybullet.URDF_USE_SELF_COLLISION):
    body_id = scene.client.loadURDF(path, [0, 0, 0], flags=flags)
    return Robot(body_id, scene)
