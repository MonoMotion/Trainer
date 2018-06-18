import os
import atexit
import signal
from subprocess import Popen
import argparse
import urllib.request
from pathlib import Path

import pybullet

import gym
from yamaxenv import YamaXEnv
from humanoid import Humanoid
from discord_reporter import DiscordReporter
from plot import plot

from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common import tf_util as U
from baselines import logger

import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--robot', type=str, default="yamax.urdf", help="Path to robot model file")
    parser.add_argument('-e', '--episodes', type=int, default=0, help="Max Number of episodes (0: unlimited)")
    parser.add_argument('-t', '--timesteps', type=int, default=0, help="Max Number of timesteps (0: unlimited)")
    parser.add_argument('-i', '--iteration', type=int, default=0, help="Max Number of iteration (0: unlimited)")
    parser.add_argument('-ss', '--seconds', type=int, default=0, help="Max Seconds (0: unlimited)")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=10000, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--schedule', type=str, default="linear", help="annealing for stepsize parameters ('constant' or 'linear')")
    parser.add_argument('--tensorboard', type=str, help="Store logs for tensorboard in this dir")
    parser.add_argument('-m', '--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=10000, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--frame-delay', type=float, default=0.0, help="Delay between each frame (for viewing result; 0 = disabled)")
    parser.add_argument('-v', '--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('--render-w', type=int, default=320, help="Rendering width")
    parser.add_argument('--render-h', type=int, default=240, help="Rendering height")
    parser.add_argument('--discord', action='store_true', default=False, help="Enable discord reporting")

    args = parser.parse_args()

    if args.tensorboard:
        if os.environ.get('OPENAI_LOG_FORMAT', 'tensorboard') != 'tensorboard':
            logger.warn('Overwriting OPENAI_LOG_FORMAT to \'tensorboard\', which was \'{}\''.format(os.environ['OPENAI_LOG_FORMAT']))
        os.environ['OPENAI_LOG_FORMAT'] = 'tensorboard'
        if os.environ.get('OPENAI_LOGDIR', args.tensorboard) != args.tensorboard:
            logger.warn('Overwriting OPENAI_LOGDIR to \'{}\', which was \'{}\''.format(args.tensorboard, os.environ['OPENAI_LOGDIR']))
        os.environ['OPENAI_LOGDIR'] = args.tensorboard
        logger.warn('Launching Tensorboard...')
        tensorboard_pid = Popen(['tensorboard', '--logdir', args.tensorboard]).pid
        def kill_tb():
            os.kill(tensorboard_pid, signal.SIGTERM)
        atexit.register(kill_tb)

    logger.configure()
    sess = U.make_session()
    sess.__enter__()

    if args.monitor:
        if not os.path.isdir(args.monitor):
            try:
                os.mkdir(args.monitor, 0o755)
            except OSError:
                raise OSError("Cannot save logs to dir {} ()".format(args.monitor))
        if args.discord:
            logger.warn('Launching Progress Reporter...')
            reporter_pid = Popen(['python3', 'discord_reporter.py', str(os.getpid())]).pid
            def kill_reporter():
                os.kill(reporter_pid, signal.SIGTERM)
            atexit.register(kill_reporter)

    if args.discord:
        reporter = DiscordReporter()
        reporter.start()
        if args.tensorboard:
            with urllib.request.urlopen('http://inet-ip.info/ip') as response:
                ip = response.read()
            reporter.report('Tensorboard is running: http://{}:6006/'.format(ip.decode()))
    else:
        reporter = None

    pybullet.connect(pybullet.GUI if args.visualize else pybullet.DIRECT)
    robot = Humanoid(urdf=args.robot, bullet_client=pybullet)

    env = YamaXEnv(robot=robot, logdir=args.monitor, renders=args.visualize, frame_delay=args.frame_delay, render_size=(args.render_w, args.render_h))

    if args.monitor:
        if args.monitor_video == 0:
            video_callable = False
        else:
            video_callable = (lambda x: x % args.monitor_video == 0)
        env = gym.wrappers.Monitor(env, args.monitor, force=not args.monitor_safe, video_callable=video_callable)

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))

    if args.save:
        if not os.path.isdir(args.save):
            try:
                os.mkdir(args.save, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(args.save))

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    def callback(l, g):
        if l["iters_so_far"] == 0:
            if reporter:
                reporter.report("Started learning.")
            if os.environ.get("OPENAI_LOG_FORMAT") == "tensorboard":
                tf.summary.FileWriter(os.path.join(os.environ.get("OPENAI_LOGDIR", "tf_logs"), "graph"), sess.graph)
            if args.load:
                tf.train.Saver().restore(sess, args.load)
        elif args.save and args.save_episodes:
            if l["episodes_so_far"] % args.save_episodes == 0:
                if reporter:
                    reporter.report("Episode {}. Saving to model...".format(l["episodes_so_far"]))
                tf.train.Saver().save(sess, "{}/afterEpisode_{}".format(args.save, l["episodes_so_far"]))

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=args.timesteps,
            max_episodes=args.episodes,
            max_iters=args.iteration,
            max_seconds=args.seconds,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule=args.schedule,
            callback=callback
        )
    env.close()
    if args.monitor:
        plot(str(Path(args.monitor).glob("*.csv").__next__()), "reward_sum", "final_distance", 10, 100, title="Plot", filename=os.path.join(args.monitor, "plot.png"))
    if args.save:
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(args.save, "final"))
    if reporter:
        reporter.report("Done!")
        if args.monitor:
            reporter.send_file(os.path.join(args.monitor, "plot.png"))
        reporter.exit()

if __name__ == '__main__':
    main()
