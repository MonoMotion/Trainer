import flom


import trainer
from trainer import simulation
from .silver_bullet import Scene, Robot
from pybullet_utils.bullet_client import BulletClient
import pybullet

import dataclasses
import logging
from colorlog import ColoredFormatter
from typing import Union, Optional


def configure_logger(raw_level: Union[int, str], log_file: Optional[str] = None):
    if isinstance(raw_level, str):
        level = getattr(logging, raw_level.upper(), None)
    else:
        level = raw_level

    if not isinstance(level, int):
        raise ValueError(f'Invalid log level: {raw_level}')

    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)

    fmt = ColoredFormatter('%(green)s%(asctime)s %(blue)s%(name)s[%(process)d] %(log_color)s%(levelname)-8s %(message)s')
    handler.setFormatter(fmt)

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)


@dataclasses.dataclass
class Trainer:
    motion: dataclasses.InitVar[str]
    robot: str
    timestep: float = 0.0165/4
    frame_skip: int = 4

    log_level: dataclasses.InitVar[str] = 'INFO'
    log_file: dataclasses.InitVar[str] = None

    _motion: flom.Motion = dataclasses.field(init=False)
    _scene: Scene = dataclasses.field(init=False)
    _robot: Robot = dataclasses.field(init=False)

    def __post_init__(self, motion, log_level, log_file):
        configure_logger(log_level, log_file)

        self._motion = flom.load(motion)
        self._scene = Scene(self.timestep, self.frame_skip)
        self._load_robot()

    def _load_robot(self):
        self._robot = simulation.reset(self._scene, self.robot)

    def train(self, output, **kwargs):
        def make_scene(_):
            gui_client = BulletClient(connection_mode=pybullet.DIRECT)
            scene = Scene(self.timestep, self.frame_skip, client=gui_client)
            robot = simulation.reset(scene, self.robot)
            return scene, robot
        trained = trainer.train(self._motion, make_scene, **kwargs)
        trained.dump(output)

    def preview(self):
        gui_client = BulletClient(connection_mode=pybullet.GUI)
        self._scene = Scene(self.timestep, self.frame_skip, client=gui_client)
        self._load_robot()
        trainer.preview(self._scene, self._motion, self._robot)

    def evaluate(self, loop=2, **kwargs):
        from trainer import evaluation
        score = evaluation.evaluate(self._scene, self._motion, self._robot, **kwargs)
        print(score)


# TODO: Move these utilities to the new package
@dataclasses.dataclass
class Utility:
    motion: dataclasses.InitVar[str]
    output: str

    log_level: dataclasses.InitVar[str] = 'INFO'
    log_file: dataclasses.InitVar[str] = None

    input_motion: flom.Motion = dataclasses.field(init=False)

    def __post_init__(self, motion, log_level, log_file):
        configure_logger(log_level, log_file)

        self.input_motion = flom.load(motion)

    def add_noise(self, random=0.1):
        trainer.utils.add_noise(self.input_motion, random)
        self.input_motion.dump(self.output)

    def plot(self, fps=0.01, loop=1):
        from trainer import plot
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plot.plot_frames(self.input_motion, ax, loop, fps)

        ax.legend()
        plt.savefig(self.output)
