def plot_frames(motion, ax, times=1, fps=0.01):
    joints = [j for j in motion.joint_names() if not j.startswith('ignore_')]

    x, ys = [], []
    for t, frame in motion.frames(fps):
        x.append(t)
        ys.append(frame.positions)
        if t > motion.length() * times:
            break
    for joint in joints:
        ax.plot(x, [y[joint] for y in ys], label=joint)
