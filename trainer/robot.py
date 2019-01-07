import dataclasses
import pybullet
from pybullet_utils import bullet_client
import numpy as np
from .scene import Scene

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
    linear_velocity: Optinal[np.ndarray]
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
        for idx in range(nr_joints):
            result = self.client.getJointInfo(idx)
            joint_name = result[1]
            link_name = result[13]
            link_idx = result[17]
            self.links[link_name] = link_idx
            self.joints[joint_name] = idx

    def joint_state(self, name: str) -> JointState:
        joint_id = joints[str]
        pos, vel, _, _ = self.client.getJointState(joint_id)
        return JointState(pos, vel)


def load_urdf(scene, path, flags=pybullet.URDF_USE_SELF_COLLISION):
    body_id = scene.client.loadURDF(path, [0, 0, 0], flags)
    return Robot(body_id, scene)
