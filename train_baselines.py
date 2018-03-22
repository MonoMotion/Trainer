#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
from yamaxenv import YamaXEnv

from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common import tf_util as U
from baselines import bench
from baselines import logger

import tensorflow as tf

import pybullet as p

def main():
    logger.configure()
    sess = U.make_session()
    sess.__enter__()
    logger.configure()
    env = bench.Monitor(YamaXEnv(logdir="./monitor/", renders=False), "./monitor/monitor.json")
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    act = pposgd_simple.learn(env, policy_fn,
            max_timesteps=10000000,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()
    saver = tf.train.Saver()
    saver.save(sess, './model/deepl2_10000000')

if __name__ == '__main__':
    main()
