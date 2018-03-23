# deepl2-pybullet-locomotion

An experiment to make [YamaX](https://y-modify.org/yamax) walk in the simulation environment using Bullet physics engine.

I used [PPO](https://arxiv.org/abs/1707.06347) (proximal policy optimization) algorithm implementation in [tensorforce](https://github.com/reinforceio/tensorforce).

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
# Train 1000000 episodes
python train.py -a config/ppo.json -n config/mlp2_network.json -e 1000000 --monitor monitor/ --save model/
# If you wanna see what is going on in simulation, add --visualize flag:
# python train.py -a config/ppo.json -n config/mlp2_network.json -e 1000000 --monitor monitor/ --save model/ --visualize
```

### Plot results

```
python plot.py monitor/log.csv reward_sum final_distance 10 100
```

### Play with trained model

```
python train.py -a config/ppo.json -n config/mlp2_network.json -e 10 --load model/ --visualize
```
