# deepl2-pybullet-locomotion

An experiment to make [YamaX](https://y-modify.org/yamax) walk in the simulation environment using Bullet physics engine.

I used [PPO](https://arxiv.org/abs/1707.06347) (proximal policy optimization) algorithm in OpenAI Baselines.

DeepL2 Project: [https://blog.y-modify.org/2018/01/04/deepl2-start/](https://blog.y-modify.org/2018/01/04/deepl2-start/)

## Try

```
git clone --recursive https://github.com/Y-modify/deepl2-pybullet-locomotion.git
cd deepl2-pybullet-locomotion
pipenv install
pipenv run python train.py log.csv
# Train for while and stop with ^C
pipenv run python plot.py log.csv
```

if baselines installation fails, try:

```
cd baselines
pipenv install -e .
cd ..
pipenv install
```
