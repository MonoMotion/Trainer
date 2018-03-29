# deepl2-pybullet-locomotion

An experiment to make [YamaX](https://y-modify.org/yamax) walk in the simulation environment using Bullet physics engine.

I used [PPO](https://arxiv.org/abs/1707.06347) (proximal policy optimization) algorithm implementation in [OpenAI Baselines](https://github.com/openai/baselines).

DeepL2 Project: [https://blog.y-modify.org/2018/01/04/deepl2-start/](https://blog.y-modify.org/2018/01/04/deepl2-start/)

## Try

### Preparation

```
git clone https://github.com/Y-modify/deepl2-pybullet-locomotion.git
cd deepl2-pybullet-locomotion
pipenv install
pipenv shell
```

### Train

```
# You can enable tensorboard to visualize the progress of learning
# export OPENAI_LOG_FORMAT=tensorboard
# export OPENAI_LOGDIR=/some/path
# tensorboard --logdir /some/path &

# Train 1000000 episodes
python train.py --monitor monitor --save model --save-episodes 500 --monitor-video 5000 --timesteps 10000000 &

# If you wanna see what is going on in simulation, add --visualize flag:
# python train.py --monitor monitor --save model --save-episodes 500 --monitor-video 5000 --timesteps 10000000 --visualize &
```

### Plot results

```
python plot.py monitor/log.csv reward_sum final_distance 10 100
```

### Play with trained model

```
python train.py --timesteps 100 --load model/ --visualize &
```
