import flom

import trainer
from trainer import simulation
from .silver_bullet import Scene, Robot
from pybullet_utils.bullet_client import BulletClient
import pybullet

import dataclasses


@dataclasses.dataclass
class Trainer:
    motion: dataclasses.InitVar[str]
    robot: str
    timestep: float = 0.0165/4
    frame_skip: int = 4

    _motion: flom.Motion = dataclasses.field(init=False)
    _scene: Scene = dataclasses.field(init=False)
    _robot: Robot = dataclasses.field(init=False)

    def __post_init__(self, motion):
        self._motion = flom.load(motion)
        self._scene = Scene(self.timestep, self.frame_skip)
        self._load_robot()

    def _load_robot(self):
        self._robot = simulation.reset(self._scene, self.robot)

    def train(self, output, chunk_length=3, num_iteration=1000, num_chunk=50, weight_factor=0.01):
        trained = trainer.train(self._scene, self._motion, self._robot, chunk_length, num_iteration, num_chunk, weight_factor)
        trained.dump(output)

    def preview(self):
        gui_client = BulletClient(connection_mode=pybullet.GUI)
        self._scene = Scene(self.timestep, self.frame_skip, client=gui_client)
        self._load_robot()
        trainer.preview(self._scene, self._motion, self._robot)

    def evaluate(self, loop=2):
        from trainer import evaluation
        score = evaluation.evaluate(self._scene, self._motion, self._robot)
        print(score)


# TODO: Move these utilities to the new package
@dataclasses.dataclass
class Utility:
    motion: dataclasses.InitVar[str]
    output: str

    input_motion: flom.Motion = dataclasses.field(init=False)

    def __post_init__(self, motion):
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
