# DeepL2

An experiment to make [YamaX](https://y-modify.org/yamax) walk in the simulation environment using Bullet physics engine.

I used [PPO](https://arxiv.org/abs/1707.06347) (proximal policy optimization) algorithm implementation in [OpenAI Baselines](https://github.com/openai/baselines).

DeepL2 Project: [https://blog.y-modify.org/2018/01/04/deepl2-start/](https://blog.y-modify.org/2018/01/04/deepl2-start/)

## Try

### Preparation

```shell
# Clone this repo
git clone https://github.com/Y-modify/deepl2.git --recursive
cd deepl2

# Setup
./setup.sh
```

### Train

The scripts' usage can be found in [coord-e/rlenv](https://github.com/coord-e/rlenv#usage)

```shell
# Train 1000000 timesteps
pipenv run train yamax --alg=ppo2 --env=YamaXForwardWalk-v0 --network=mlp --num_timesteps=1e6
```

### Play with trained model

```shell
pipenv run play yamax
```
