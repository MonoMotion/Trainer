#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
import argparse
from yamaxenv import YamaXEnv

from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common import tf_util as U
from baselines import bench
from baselines import logger

import tensorflow as tf

import pybullet as p

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    # parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    # parser.add_argument('-se', '--save-episodes', type=int, default=5000, help="Save agent every x episodes")
    # parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    # parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    # parser.add_argument('--monitor-video', type=int, default=5000, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")

    args = parser.parse_args()

    logger.configure()
    sess = U.make_session()
    sess.__enter__()
    logger.configure()
    env = YamaXEnv(logdir=args.monitor, renders=args.visualize)
    if args.monitor:
        env = bench.Monitor(env, args.monitor)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    act = pposgd_simple.learn(env, policy_fn,
            max_timesteps=args.timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()
    if args.save:
        saver = tf.train.Saver()
        saver.save(sess, args.save)

if __name__ == '__main__':
    main()
