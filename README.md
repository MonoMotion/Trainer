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

`run.py` can be used in the same way as [baselines' `run.py`](https://github.com/openai/baselines/blob/115b59d28b79523826dd5a81fbc5d6f8ed431c7c/README.md#training-models)

```shell
# Train 1000000 timesteps
python run.py --alg=ppo2 --env=YamaXForwardWalk-v0 --network=mlp --num_timesteps=1e6
```

You can enable tensorboard to visualize the progress of learning

```shell
# Set them before training
export OPENAI_LOG_FORMAT='tensorboard'
export OPENAI_LOGDIR=./tblog
tensorboard --logdir=$OPENAI_LOGDIR
```

### Play with trained model

```shell
python run.py --alg=ppo2 --env=YamaXForwardWalk-v0 --num_timesteps=0 --load_path=./models/model --play
```
