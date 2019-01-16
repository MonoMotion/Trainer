# DeepL2.Trainer
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FDeepL2%2FTrainer.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FDeepL2%2FTrainer?ref=badge_shield)


An experiment to make [YamaX](https://y-modify.org/yamax) walk in the simulation environment using Bullet physics engine.

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

```shell
pipenv run train --motion input.fom --robot robot.urdf --output trained.fom
```

### Preview the trained motion

```shell
pipenv run preview --motion trained.fom --robot robot.urdf
```

## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FDeepL2%2FTrainer.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FDeepL2%2FTrainer?ref=badge_large)
