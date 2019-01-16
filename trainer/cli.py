import flom

import trainer

import dataclasses


@dataclasses.dataclass
class Trainer:
    motion: dataclasses.InitVar[str]
    robot: str
    timestep: float = 0.0165/4
    frame_skip: int = 4

    input_motion: flom.Motion = dataclasses.field(init=False)

    def __post_init__(self, motion):
        self.input_motion = flom.load(motion)

    def train(self, output, chunk_length=3, num_iteration=1000, num_chunk=50, weight_factor=0.01):
        trained = trainer.train(self.input_motion, self.robot, self.timestep,
                                self.frame_skip, chunk_length, num_iteration, num_chunk, weight_factor)
        trained.dump(output)

    def preview(self):
        trainer.preview(self.input_motion, self.robot, self.timestep, self.frame_skip)


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
