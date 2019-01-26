import flom


import trainer
from trainer import simulation
from silverbullet import Scene, Robot
from silverbullet.connection import Mode, Connection
from pybullet_utils.bullet_client import BulletClient
import pybullet
import numpy as np

import dataclasses
import logging
from colorlog import ColoredFormatter
from typing import Union, Optional
import random


log = logging.getLogger(__name__)

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
class CLI:
    motion: dataclasses.InitVar[str]
    robot: str
    timestep: float = 0.0165/4
    frame_skip: int = 4

    seed: Optional[int] = None

    save_snapshot: Optional[int] = None  # save snapshot every x times
    snapshot_pattern: str = 'snapshot{}.fom'

    log_level: dataclasses.InitVar[str] = 'INFO'
    log_file: dataclasses.InitVar[str] = None

    _motion: flom.Motion = dataclasses.field(init=False)
    _scene: Scene = dataclasses.field(init=False)
    _robot: Robot = dataclasses.field(init=False)

    def __post_init__(self, motion, log_level, log_file):
        configure_logger(log_level, log_file)

        self._seed()

        self._motion = flom.load(motion)
        self._scene = Scene(self.timestep, self.frame_skip)
        self._load_robot()

    def _load_robot(self):
        self._robot = simulation.reset(self._scene, self.robot)

    def _seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def train(self, output, **kwargs):
        def snapshot(chunk_idx, build_motion):
            if chunk_idx % self.save_snapshot != 0:
                return
            path = self.snapshot_pattern.format(chunk_idx)

            log.info(f'Saving snapshot to {path}')
            build_motion().dump(path)

        callback = snapshot if self.save_snapshot else None
        trained = trainer.train(self._scene, self._motion, self._robot, callback=callback, **kwargs)
        trained.dump(output)

    def preview(self, real_time=False):
        conn = Connection(mode=Mode.GUI)
        self._scene = Scene(self.timestep, self.frame_skip, connection=conn)
        self._load_robot()
        trainer.preview(self._scene, self._motion, self._robot, real_time)

    def evaluate(self, loop=2, **kwargs):
        from trainer import evaluation
        score = evaluation.evaluate(self._scene, self._motion, self._robot, **kwargs)
        print(score)
