import flom

import trainer

def train(motion, output, robot, timestep=0.0165/4, frame_skip=4, chunk_length=3, num_iteration=1000, num_chunk=50, weight_factor=0.01):
    m = flom.load(motion)
    trained = trainer.train(m, robot, timestep, frame_skip, chunk_length, num_iteration, num_chunk, weight_factor)
    trained.dump(output)

def preview(motion, robot, timestep=0.0165/4, frame_skip=4):
    m = flom.load(motion)
    trainer.preview(m, robot, timestep, frame_skip)

def add_noise(motion, output, random=0.1):
    m = flom.load(motion)
    trainer.utils.add_noise(m, random)
    m.dump(output)

def plot(motion, output, fps=0.01, loop=1):
    from trainer import plot
    import matplotlib.pyplot as plt

    m = flom.load(motion)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plot.plot_frames(m, ax, loop, fps)

    ax.legend()
    plt.savefig(output)
