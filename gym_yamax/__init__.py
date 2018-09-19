from gym.envs.registration import register

register(
        id='YamaXForwardWalk-v0',
        entry_point='gym_yamax:RoboschoolYamaXForwardWalk',
        max_episode_steps=1000
        )

from gym_yamax.yamax_forward_walk import RoboschoolYamaXForwardWalk
