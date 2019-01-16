from pybullet_utils import bullet_client
import pybullet_data
import pybullet

import dataclasses
from typing import Sequence, Optional, Union

from .color import Color


@dataclasses.dataclass(frozen=True)
class DebugBody:
    body_id: int

    def remove_from_scene(self, scene):
        # TODO: annotate scene with Scene type
        scene.client.removeBody(self.body_id)


@dataclasses.dataclass(frozen=True)
class DebugSphere(DebugBody):
    pass


@dataclasses.dataclass(frozen=True)
class DebugItem:
    item_id: int

    def remove_from_scene(self, scene):
        # TODO: annotate scene with Scene type
        scene.client.removeUserDebugItem(self.item_id)


@dataclasses.dataclass(frozen=True)
class DebugLine(DebugItem):
    pass


@dataclasses.dataclass(frozen=True)
class DebugText(DebugItem):
    pass


@dataclasses.dataclass(frozen=True)
class SavedState:
    state_id: int


@dataclasses.dataclass
class Scene:
    timestep: float
    frame_skip: int
    gravity: float = 9.8
    client: bullet_client.BulletClient = None

    dt: float = dataclasses.field(init=False)
    ts: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.dt = self.timestep * self.frame_skip

        self.connect(self.client)
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
        self.ts = 0
        self.load_plane()

    def step(self):
        self.client.stepSimulation()
        self.ts += self.dt

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

    def draw_text(self, text: str, pos: Sequence[float], orientation: Optional[Sequence[float]] = None, color: Optional[Color] = None, size: Optional[float] = None, replace: Optional[DebugItem] = None) -> DebugText:
        args = {
            'text': text,
            'textPosition': pos,
        }

        if color is not None:
            args['textColorRGB'] = color.as_rgb()

        if orientation is not None:
            args['textOrientation'] = orientation

        if replace is not None:
            args['replaceItemUniqueId'] = replace.item_id

        if size is not None:
            args['textSize'] = size

        item_id = self.client.addUserDebugText(**args)
        return DebugText(item_id)

    def draw_sphere(self, pos: Sequence[float], radius: Optional[float] = None, color: Optional[Color] = None) -> DebugSphere:
        args = {
            'shapeType': pybullet.GEOM_SPHERE
        }

        if color is not None:
            args['rgbaColor'] = color.as_rgba()

        if radius is not None:
            args['radius'] = radius

        visual_id = self.client.createVisualShape(**args)
        body_id = self.client.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=pos)
        return DebugSphere(body_id)

    def move_debug_body(self, body: DebugBody, position: Sequence[float], orientation: Sequence[float]):
        self.client.resetBasePositionAndOrientation(body.body_id, position, orientation)

    def remove_debug_object(self, o: Union[DebugItem, DebugBody]):
        o.remove_from_scene(self)

    def save_state(self) -> SavedState:
        state_id = self.client.saveState()
        return SavedState(state_id)

    def restore_state(self, state: SavedState):
        self.client.restoreState(state.state_id)
