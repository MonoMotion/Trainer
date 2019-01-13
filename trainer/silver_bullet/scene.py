from pybullet_utils import bullet_client
import pybullet_data

import dataclasses
from typing import Sequence, Optional

from .color import Color

@dataclasses.dataclass
class DebugItem:
    item_id: int

@dataclasses.dataclass
class DebugLine(DebugItem):
    pass

class Scene(object):
    def __init__(self, gravity, timestep, frame_skip, client=None):
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.dt = timestep * frame_skip

        self.connect(client)
        self.episode_restart()

    def connect(self, client):
        if client:
            self.client = client
        else:
            self.client = bullet_client.BulletClient()

    def clean_everything(self):
        self.client.resetSimulation()
        self.configure_simulation()

    def configure_simulation(self):
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -self.gravity)
        self.client.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSubSteps=self.frame_skip)

    def load_plane(self):
        self.plane_id = self.client.loadURDF("plane.urdf")

    def episode_restart(self):
        self.clean_everything()
        self.load_plane()

    def step(self):
        self.client.stepSimulation()

    def draw_line(self, from_pos: Sequence[float], to_pos: Sequence[float], color: Optional[Color] = None, width: Optional[float] = None, replace: Optional[DebugItem] = None) -> DebugLine:
        args = {
            'lineFromXYZ': from_pos,
            'lineToXYZ': to_pos
        }

        if color is not None:
            args['lineColorRGB'] = color.as_rgb()

        if replace is not None:
            args['replaceItemUniqueId'] = replace.item_id

        if width is not None:
            args['lineWidth'] = width

        item_id = self.client.addUserDebugLine(**args)
        return DebugLine(item_id)
