# DeepL2

An experiment to make [YamaX](https://y-modify.org/yamax) walk in the simulation environment using Bullet physics engine.

I used [PPO](https://arxiv.org/abs/1707.06347) (proximal policy optimization) algorithm implementation in [OpenAI Baselines](https://github.com/openai/baselines).

DeepL2 Project: [https://blog.y-modify.org/2018/01/04/deepl2-start/](https://blog.y-modify.org/2018/01/04/deepl2-start/)

## Try

### Preparation

```shell
# Clone this repo
git clone https://github.com/Y-modify/deepl2.git
cd deepl2

# Setup
./setup.sh

# Activate the environment
pipenv shell
```

### Train

```shell
# Train 1000000 episodes
python train.py --monitor monitor --save model --save-episodes 5000 --monitor-video 5000 --timesteps 10000000 &

# You can enable tensorboard to visualize the progress of learning
# python train.py --monitor monitor --save model --save-episodes 5000 --monitor-video 5000 --timesteps 10000000 --tensorboard ./tblog &

# If you wanna see what is going on in simulation, add --visualize flag:
# python train.py --monitor monitor --save model --save-episodes 5000 --monitor-video 5000 --timesteps 10000000 --visualize &
```

### Plot results

```shell
python plot.py monitor/log.csv reward_sum final_distance 10 100
```

### Play with trained model

```shell
python train.py --timesteps 100 --load model/final --visualize &
```
